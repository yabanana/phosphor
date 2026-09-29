#include "imgui/imgui_renderer.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/pipeline_cache.h"
#include "core/log.h"

#include <imgui.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace phosphor {

namespace {

NS::String* str(const char* s) {
    return NS::String::string(s, NS::UTF8StringEncoding);
}

constexpr size_t kAlign = 256;

size_t alignUp(size_t v) { return (v + kAlign - 1) & ~(kAlign - 1); }

// Argument table slots; must match shaders/imgui.metal.
enum Binding : NS::UInteger {
    BindVertices = 0,
    BindUniforms = 1,
    BindCount    = 2,
    BindImage    = 0, // texture slot
};

struct ImGuiUniforms {
    float projection[16]; // column-major
};

static_assert(sizeof(ImDrawVert) == 20, "ImGuiVertex in imgui.metal mirrors ImDrawVert");
static_assert(sizeof(ImTextureID) == sizeof(MTL::ResourceID), "texture IDs carry MTL::ResourceID");

} // namespace

ImGuiRenderer::ImGuiRenderer(MetalContext& context, PipelineCache& pipelines)
    : context_(context), pipelines_(pipelines) {
    ImGuiIO& io = ImGui::GetIO();
    io.BackendRendererName = "phosphor_mtl4";
    io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;

    buildPipeline();

    // The overlay shares the scene pass, whose depth attachment is bound:
    // draw on top without testing or writing depth.
    MTL::DepthStencilDescriptor* dsDesc = MTL::DepthStencilDescriptor::alloc()->init();
    dsDesc->setDepthCompareFunction(MTL::CompareFunctionAlways);
    dsDesc->setDepthWriteEnabled(false);
    depthState_ = context_.device()->newDepthStencilState(dsDesc);
    dsDesc->release();

    NS::Error* error = nullptr;
    MTL4::ArgumentTableDescriptor* atDesc = MTL4::ArgumentTableDescriptor::alloc()->init();
    atDesc->setMaxBufferBindCount(BindCount);
    atDesc->setMaxTextureBindCount(1);
    atDesc->setLabel(str("ImGui arguments"));
    arguments_ = context_.device()->newArgumentTable(atDesc, &error);
    atDesc->release();
    if (!arguments_) {
        throw std::runtime_error("Failed to create ImGui argument table");
    }

    createFontTexture();
}

ImGuiRenderer::~ImGuiRenderer() {
    context_.memory().release(fontTexture_, MemoryCategory::Textures);
    arguments_->release();
    depthState_->release();

    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->SetTexID(ImTextureID{});
    io.BackendRendererName = nullptr;
    io.BackendFlags &= ~ImGuiBackendFlags_RendererHasVtxOffset;
}

void ImGuiRenderer::buildPipeline() {
    pipe::PipelineDesc desc;
    desc.label        = "ImGui";
    desc.functions[0] = "imgui_vs";
    desc.functions[1] = "imgui_fs";
    desc.output(0, rg::Format::BGRA8Srgb, pipe::ColorOutput::Blend::AlphaOver);
    // Created before the first frame: the engine waits for every pipeline.
    pipeline_ = pipelines_.request(desc);
}

void ImGuiRenderer::createFontTexture() {
    ImGuiIO& io = ImGui::GetIO();
    unsigned char* pixels = nullptr;
    int width = 0, height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatRGBA8Unorm, static_cast<NS::UInteger>(width), static_cast<NS::UInteger>(height), false);
    desc->setUsage(MTL::TextureUsageShaderRead);
    desc->setStorageMode(MTL::StorageModePrivate);
    fontTexture_ = context_.memory().newTexture(desc, MemoryCategory::Textures, "ImGui font atlas");

    const size_t rowBytes = static_cast<size_t>(width) * 4;
    const UploadRing::Slice staging = context_.stagingAllocate(rowBytes * static_cast<size_t>(height));
    std::memcpy(staging.cpu, pixels, rowBytes * static_cast<size_t>(height));
    // Blit upload keeps the private texture eligible for lossless compression (O6).
    MTL::Texture* font = fontTexture_;
    context_.enqueueUpload([staging, rowBytes, width, height, font](MTL4::ComputeCommandEncoder* enc) {
        enc->copyFromBuffer(staging.buffer, staging.offset, rowBytes, 0,
                            MTL::Size::Make(static_cast<NS::UInteger>(width), static_cast<NS::UInteger>(height), 1),
                            font, 0, 0, MTL::Origin::Make(0, 0, 0));
    });
    context_.flushUploads();

    io.Fonts->SetTexID(static_cast<ImTextureID>(fontTexture_->gpuResourceID()._impl));
}

void ImGuiRenderer::setupRenderState(MTL4::RenderCommandEncoder* encoder, MTL::GPUAddress vertices, MTL::GPUAddress uniforms) {
    arguments_->setAddress(vertices, BindVertices);
    arguments_->setAddress(uniforms, BindUniforms);

    encoder->setRenderPipelineState(pipelines_.render(pipeline_));
    encoder->setDepthStencilState(depthState_);
    encoder->setArgumentTable(arguments_, MTL::RenderStageVertex | MTL::RenderStageFragment);
    // Viewport (full target) and CullModeNone are Metal's defaults, and the
    // scene pass fused before this one leaves them so (render graph, F2.7);
    // setting them again is flagged as redundant by the validation layer.
}

void ImGuiRenderer::render(MTL4::RenderCommandEncoder* encoder, const ImDrawData* drawData) {
    if (!drawData || drawData->CmdListsCount == 0 || drawData->TotalVtxCount == 0) return;

    const float fbWidth  = drawData->DisplaySize.x * drawData->FramebufferScale.x;
    const float fbHeight = drawData->DisplaySize.y * drawData->FramebufferScale.y;
    if (fbWidth <= 0.0f || fbHeight <= 0.0f) return;

    // --- Per-frame upload: projection, vertices, indices --------------------
    const size_t vertexBytes    = static_cast<size_t>(drawData->TotalVtxCount) * sizeof(ImDrawVert);
    const size_t indexBytes     = static_cast<size_t>(drawData->TotalIdxCount) * sizeof(ImDrawIdx);
    const size_t verticesOffset = alignUp(sizeof(ImGuiUniforms));
    const size_t indicesOffset  = verticesOffset + alignUp(vertexBytes);
    const size_t totalSize      = indicesOffset + alignUp(indexBytes);

    const UploadRing::Slice upload = context_.frameUploads().allocate(totalSize);
    u8* base = upload.cpu;

    // Orthographic projection from ImGui's top-left origin to NDC (y up), z = 0.
    const float l = drawData->DisplayPos.x;
    const float r = drawData->DisplayPos.x + drawData->DisplaySize.x;
    const float t = drawData->DisplayPos.y;
    const float b = drawData->DisplayPos.y + drawData->DisplaySize.y;
    const ImGuiUniforms uniforms{{
        2.0f / (r - l),    0.0f,              0.0f, 0.0f,
        0.0f,              2.0f / (t - b),    0.0f, 0.0f,
        0.0f,              0.0f,              1.0f, 0.0f,
        (r + l) / (l - r), (t + b) / (b - t), 0.0f, 1.0f,
    }};
    std::memcpy(base, &uniforms, sizeof(uniforms));

    size_t vtxCursor = 0;
    size_t idxCursor = 0;
    for (const ImDrawList* list : drawData->CmdLists) {
        std::memcpy(base + verticesOffset + vtxCursor * sizeof(ImDrawVert), list->VtxBuffer.Data,
                    static_cast<size_t>(list->VtxBuffer.Size) * sizeof(ImDrawVert));
        std::memcpy(base + indicesOffset + idxCursor * sizeof(ImDrawIdx), list->IdxBuffer.Data,
                    static_cast<size_t>(list->IdxBuffer.Size) * sizeof(ImDrawIdx));
        vtxCursor += static_cast<size_t>(list->VtxBuffer.Size);
        idxCursor += static_cast<size_t>(list->IdxBuffer.Size);
    }

    // --- Draws -------------------------------------------------------------
    const MTL::GPUAddress bufferBase = upload.gpu;
    const MTL::GPUAddress vertices   = bufferBase + verticesOffset;
    const MTL::GPUAddress indices    = bufferBase + indicesOffset;
    setupRenderState(encoder, vertices, bufferBase);

    const ImVec2 clipOffset = drawData->DisplayPos;
    const ImVec2 clipScale  = drawData->FramebufferScale;
    ImTextureID boundTexture{};
    vtxCursor = 0;
    idxCursor = 0;
    for (const ImDrawList* list : drawData->CmdLists) {
        for (const ImDrawCmd& cmd : list->CmdBuffer) {
            if (cmd.UserCallback) {
                if (cmd.UserCallback == ImDrawCallback_ResetRenderState) {
                    setupRenderState(encoder, vertices, bufferBase);
                    boundTexture = ImTextureID{};
                } else {
                    cmd.UserCallback(list, &cmd);
                }
                continue;
            }

            // Clip rectangle in framebuffer pixels, clamped: scissors must lie inside the target.
            const float x0 = std::max((cmd.ClipRect.x - clipOffset.x) * clipScale.x, 0.0f);
            const float y0 = std::max((cmd.ClipRect.y - clipOffset.y) * clipScale.y, 0.0f);
            const float x1 = std::min((cmd.ClipRect.z - clipOffset.x) * clipScale.x, fbWidth);
            const float y1 = std::min((cmd.ClipRect.w - clipOffset.y) * clipScale.y, fbHeight);
            if (x1 <= x0 || y1 <= y0 || cmd.ElemCount == 0) continue;

            encoder->setScissorRect(MTL::ScissorRect{static_cast<NS::UInteger>(x0), static_cast<NS::UInteger>(y0),
                                                     static_cast<NS::UInteger>(x1 - x0),
                                                     static_cast<NS::UInteger>(y1 - y0)});

            const ImTextureID texture = cmd.GetTexID();
            if (texture != boundTexture) {
                arguments_->setTexture(MTL::ResourceID{static_cast<uint64_t>(texture)}, BindImage);
                boundTexture = texture;
            }

            const size_t firstIndex = idxCursor + cmd.IdxOffset;
            encoder->drawIndexedPrimitives(
                MTL::PrimitiveTypeTriangle, cmd.ElemCount,
                sizeof(ImDrawIdx) == 2 ? MTL::IndexTypeUInt16 : MTL::IndexTypeUInt32,
                indices + firstIndex * sizeof(ImDrawIdx), static_cast<NS::UInteger>(cmd.ElemCount) * sizeof(ImDrawIdx),
                1, static_cast<NS::Integer>(vtxCursor + cmd.VtxOffset), 0);
        }
        vtxCursor += static_cast<size_t>(list->VtxBuffer.Size);
        idxCursor += static_cast<size_t>(list->IdxBuffer.Size);
    }
}

} // namespace phosphor
