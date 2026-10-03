#pragma once

#include "core/types.h"

#include <functional>
#include <string>
#include <vector>

namespace phosphor::rg {

// ---------------------------------------------------------------------------
// Render graph (F2) -- portable description of a frame as passes that read
// and write virtual resources.  No Metal types here: the backend
// (src/platform/metal/metal_graph_executor) executes a CompiledGraph.
//
// Accesses are versioned: every write of a resource creates a new version,
// and a read names the version it consumes.  Dependencies therefore follow
// the data, not declaration order:
//   RAW  reader of v        after the writer of v
//   WAW  writer of v+1      after the writer of v
//   WAR  writer of v+1      after every reader of v
// A version can be written only once (the version chain of a resource is
// linear); writing a stale handle is a setup error.
//
// The graph is built rarely (resize, bench switch) and compiled once; the
// per-frame work is only the execution of the cached CompiledGraph (O7).
// Imported resources (drawable, persistent buffers) are bound to physical
// objects per frame by the executor.
// ---------------------------------------------------------------------------

using Stages = u32;
// Pipeline stages of an access (mirror MTL::Stages; the backend maps them).
enum Stage : Stages {
    StageNone = 0,
    StageVertex = 1u << 0,
    StageFragment = 1u << 1,
    StageTile = 1u << 2,
    StageObject = 1u << 3,
    StageMesh = 1u << 4,
    StageDispatch = 1u << 5, // compute kernels
    StageBlit = 1u << 6,     // copies, fills, mipmap generation
    StageAccelerationStructure = 1u << 7,
    StageMachineLearning = 1u << 8,
    StageExternal =
        StageVertex | StageFragment | StageObject | StageMesh | StageDispatch | StageBlit | StageMachineLearning,
    // Raster stages that run before rasterisation (geometry front end).
    StageGeometry = StageVertex | StageObject | StageMesh,
    StageRaster = StageGeometry | StageFragment | StageTile,
};

enum class Queue : u8 {
    Graphics,     // the frame's main queue
    AsyncCompute, // second queue (F2.6); synchronised with events
};

// Encoder type a pass records into.
enum class PassType : u8 {
    Raster,   // render encoder; may be fused with adjacent raster passes (F2.4)
    Compute,  // compute encoder (MTL4 compute encoders also do copies)
    Blit,     // copies only; recorded like Compute on Metal 4
    External, // framework owns encoders (MetalFX); explicit graph accesses + executor fence boundary
};

enum class ResourceKind : u8 { Texture, Buffer };

// Abstract texture formats (the backend maps them to MTL::PixelFormat).
enum class Format : u16 {
    Unknown,
    R8Unorm,
    RG8Unorm,
    RGBA8Unorm,
    RGBA8Srgb,
    BGRA8Unorm,
    BGRA8Srgb,
    R16Float,
    RG16Float,
    RGBA16Float,
    R32Float,
    RG32Float,
    RGBA32Float,
    R32Uint,
    RG11B10Float,
    RGB10A2Unorm,
    Depth16Unorm,
    Depth32Float,
    Depth32FloatStencil8,
};

[[nodiscard]] u32  bytesPerPixel(Format format);
[[nodiscard]] bool isDepthFormat(Format format);
[[nodiscard]] const char* formatName(Format format);

struct TextureDesc {
    Format format      = Format::Unknown;
    u32    width       = 0;
    u32    height      = 0;
    u32    depth       = 1; // array layers for 2D arrays
    u32    mipLevels   = 1;
    u32    sampleCount = 1;
    /// Estimated bytes of one full copy (all mips/layers/samples), for O1.
    [[nodiscard]] u64 estimatedBytes() const;
    bool operator==(const TextureDesc&) const = default;
};

struct BufferDesc {
    u64 size = 0;
    bool operator==(const BufferDesc&) const = default;
};

// How a pass uses a resource.  Decides load/store deduction and bandwidth.
enum class Usage : u8 {
    ColorAttachment,   // raster: render target (write; Preserve = blending/load)
    DepthAttachment,   // raster: depth/stencil target, read + write
    DepthRead,         // raster: read-only depth attachment (test, no write)
    ShaderRead,        // sampled texture / device buffer read in `stages`
    ShaderWrite,       // storage texture / device buffer written in `stages`
    CopySrc,           // blit source
    CopyDst,           // blit destination
    IndirectArgs,      // indirect draw/dispatch arguments
};

[[nodiscard]] bool isWrite(Usage usage);
[[nodiscard]] bool isAttachment(Usage usage);

// What an attachment write needs from the previous contents.
enum class LoadIntent : u8 {
    Clear,    // start from the clear value
    Preserve, // needs the previous contents (blend/overlay onto them)
    Discard,  // every pixel is overwritten: previous contents irrelevant
};

struct ClearValue {
    float color[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float depth    = 0.0f; // reverse-Z: far plane is 0
    u8    stencil  = 0;
};

// Handles: `resource` indexes RenderGraph::resources(), `version` is the
// version of its contents this handle refers to (0 = initial contents).
struct TextureRef {
    u32 resource = ~0u;
    u32 version  = 0;
    [[nodiscard]] bool valid() const { return resource != ~0u; }
    bool operator==(const TextureRef&) const = default;
};
struct BufferRef {
    u32 resource = ~0u;
    u32 version  = 0;
    [[nodiscard]] bool valid() const { return resource != ~0u; }
    bool operator==(const BufferRef&) const = default;
};

// Imported resource flags.
enum ImportFlags : u32 {
    ImportNone = 0,
    /// Initial contents (version 0) are meaningful and may be read (e.g. a
    /// persistent buffer).  Without it, reading version 0 is an error and a
    /// first attachment use cannot load (e.g. a fresh drawable).
    ImportContentsDefined = 1u << 0,
    /// The final version is consumed after the graph (present, readback,
    /// next frame): its writers are never culled and attachments store it.
    ImportOutput = 1u << 1,
    /// A different physical resource is bound every frame and reused only
    /// once the frame that used it has completed (drawable, per-slot
    /// readback buffers).  Without it, an imported resource written by the
    /// graph is the same memory in consecutive frames, and its first access
    /// in a frame waits for the previous frame's accesses (barrier plan).
    ImportPerFrame = 1u << 2,
};

struct Access {
    u32        resource = ~0u;
    u32        version  = 0;  // read: version consumed; write: version produced
    Usage      usage    = Usage::ShaderRead;
    Stages     stages   = StageNone;
    // Attachments only.
    u32        slot     = 0;  // color attachment index
    LoadIntent load     = LoadIntent::Discard;
    ClearValue clear;
};

struct ResourceNode {
    std::string  name;
    ResourceKind kind     = ResourceKind::Texture;
    TextureDesc  texture;
    BufferDesc   buffer;
    bool         imported = false;
    u32          importFlags = ImportNone;
    u32          versions = 1; // number of versions created so far (>= 1)
};

/// OPT-1: work of a pass for the graph cost model (optimizer/cost_model.h),
/// whole-pass totals.  Integer multiply-adds count once (one IMAD).
struct PassCost {
    double flops       = 0; // FP32, FMA = 2
    double intOps      = 0; // integer ops (IMAD = 1)
    double triangles   = 0; // rasterised triangles
    double invocations = 0; // fragments / threads (informational)
    [[nodiscard]] bool empty() const { return flops == 0 && intOps == 0 && triangles == 0; }
};

class PassContext; // defined in render_graph_exec.h (backend-facing)
using ExecuteFn = std::function<void(PassContext&)>;

// Scheduling hints (S-TBDR-6): after a fragment-heavy pass, a ready
// geometry-heavy pass is preferred so its vertex work overlaps.
enum PassHint : u32 {
    HintNone          = 0,
    HintGeometryHeavy = 1u << 0,
    HintFragmentHeavy = 1u << 1,
};

// A Preserve attachment write also appears in `reads` (version consumed)
// with usage ColorAttachment or DepthRead: an attachment load.  A
// ColorAttachment read without a write of the same resource in the pass is a
// per-pixel read of the attachment (PassBuilder::readColor); its `slot` is
// the attachment index.
struct PassNode {
    std::string name;
    PassType    type  = PassType::Raster;
    Queue       queue = Queue::Graphics;
    std::vector<Access> reads;
    std::vector<Access> writes;
    bool        sideEffect = false;
    u32         hints      = HintNone;
    /// F2.5: the pass may be encoded as `parallelChunks` independent pieces
    /// on different threads (execute receives the chunk index).
    u32         parallelChunks = 1;
    /// Fixed tile size for a raster pass containing a tile dispatch. Zero
    /// leaves the driver's choice; fused members must agree when nonzero.
    u32 tileWidth = 0, tileHeight = 0;
    /// F4: shader functions the pass runs (profiling only: maps Metal System
    /// Trace's per-shader timeline to passes; no effect on compilation).
    std::vector<std::string> profileShaders;
    /// OPT-1: work the pass declares for the graph cost model (optional:
    /// all zero -> the model counts only its DRAM bytes and fixed costs).
    PassCost    cost;
    ExecuteFn   execute;
};

class RenderGraph;

// Handed to the setup callback of addPass().
class PassBuilder {
public:
    /// New transient resources (contents undefined until first written).
    TextureRef createTexture(const std::string& name, const TextureDesc& desc);
    BufferRef  createBuffer(const std::string& name, const BufferDesc& desc);

    // --- Raster attachments (returns the new version) -----------------------
    TextureRef writeColor(TextureRef target, u32 slot, LoadIntent load, const ClearValue& clear = {});
    TextureRef writeDepth(TextureRef target, LoadIntent load, const ClearValue& clear = {});
    void       readDepth(TextureRef target);
    /// OPT-1: read color attachment `slot` per pixel without writing it
    /// (programmable blending / framebuffer fetch).  Bound as an attachment of
    /// the pass's render group: loaded unless an earlier member of the group
    /// wrote it in tile memory.
    void readColor(TextureRef target, u32 slot, Stages stages = StageFragment);
    void setTileSize(u32 width, u32 height);

    // --- Shader / copy accesses ----------------------------------------------
    void       read(TextureRef texture, Usage usage, Stages stages);
    void       read(BufferRef buffer, Usage usage, Stages stages);
    TextureRef write(TextureRef texture, Usage usage, Stages stages);
    BufferRef  write(BufferRef buffer, Usage usage, Stages stages);

    /// The pass has effects outside the graph (readback, present, UI state):
    /// it is never culled.
    void setSideEffect();
    void setHints(u32 hints);
    void setParallelChunks(u32 chunks);
    /// F4: comma-separated shader function names run by the pass (profiling).
    void setProfileShaders(const std::string& functions);
    /// OPT-1: declared work for the graph cost model.
    void setCost(const PassCost& cost);

private:
    friend class RenderGraph;
    PassBuilder(RenderGraph& graph, u32 pass) : graph_(graph), pass_(pass) {}
    RenderGraph& graph_;
    u32          pass_;
};

class RenderGraph {
public:
    using SetupFn = std::function<void(PassBuilder&)>;

    RenderGraph() = default;
    RenderGraph(const RenderGraph&) = delete;
    RenderGraph& operator=(const RenderGraph&) = delete;

    /// External resources.  `flags` is a combination of ImportFlags.
    TextureRef importTexture(const std::string& name, const TextureDesc& desc, u32 flags);
    BufferRef  importBuffer(const std::string& name, const BufferDesc& desc, u32 flags);

    /// Declare a pass; `setup` runs immediately and declares its accesses.
    /// Returns the pass index.
    u32 addPass(const std::string& name, PassType type, Queue queue, const SetupFn& setup, ExecuteFn execute);
    u32 addPass(const std::string& name, PassType type, const SetupFn& setup, ExecuteFn execute) {
        return addPass(name, type, Queue::Graphics, setup, std::move(execute));
    }

    /// Drop every pass and resource.
    void reset();

    [[nodiscard]] const std::vector<PassNode>&     passes()    const { return passes_; }
    [[nodiscard]] const std::vector<ResourceNode>& resources() const { return resources_; }
    [[nodiscard]] std::vector<PassNode>&           passes()          { return passes_; }

    /// Setup errors (stale write, read of undefined contents, wrong usage).
    /// Non-empty errors make compile() fail.
    [[nodiscard]] const std::vector<std::string>& errors() const { return errors_; }

private:
    friend class PassBuilder;
    u32  addResource(const std::string& name, ResourceKind kind, const TextureDesc& t, const BufferDesc& b,
                     bool imported, u32 flags);
    void addRead(u32 pass, u32 resource, u32 version, Usage usage, Stages stages);
    u32  addWrite(u32 pass, u32 resource, u32 version, Usage usage, Stages stages, u32 slot, LoadIntent load,
                  const ClearValue& clear);
    void error(const std::string& message);

    std::vector<PassNode>     passes_;
    std::vector<ResourceNode> resources_;
    std::vector<std::string>  errors_;
};

// ---------------------------------------------------------------------------
// Compiled graph
// ---------------------------------------------------------------------------

enum class DepKind : u8 { RAW, WAR, WAW };

struct Dependency {
    u32     from     = 0; // pass index (declaration order)
    u32     to       = 0;
    u32     resource = 0;
    DepKind kind     = DepKind::RAW;
};

// Lifetime of a resource over the execution order (positions in
// CompiledGraph::order).  Unused resources have first > last.
struct Lifetime {
    u32 first = ~0u;
    u32 last  = 0;
    [[nodiscard]] bool used() const { return first <= last; }
    [[nodiscard]] bool overlaps(const Lifetime& o) const {
        return used() && o.used() && first <= o.last && o.first <= last;
    }
};

// --- F2.4: TBDR pass fusion and attachment actions ------------------------

enum class LoadAction : u8 { DontCare, Load, Clear };
enum class StoreAction : u8 { DontCare, Store };

struct AttachmentPlan {
    u32         resource = 0;
    bool        depth    = false; // depth/stencil attachment, else color
    u32         slot     = 0;     // color slot
    LoadAction  load     = LoadAction::DontCare;
    StoreAction store    = StoreAction::DontCare;
    ClearValue  clear;
    bool        readOnly = false; // only read in the whole group (DepthRead / readColor)
};

// A render encoder: consecutive raster passes (positions in the order)
// fused into one render pass that keeps its attachments in tile memory.
struct RenderGroup {
    u32 firstPosition = 0;   // positions in CompiledGraph::order
    u32 lastPosition  = 0;   // inclusive
    u32 width = 0, height = 0, sampleCount = 1;
    u32 tileWidth = 0, tileHeight = 0;
    std::vector<AttachmentPlan> attachments;
};

// An encoder of the executed frame: one render group, or a run of
// consecutive compute/blit passes on the same queue (Metal 4 compute
// encoders also record copies), which then synchronise with intra-encoder
// barriers.
struct EncoderPlan {
    PassType type = PassType::Raster; // Raster, Compute (includes Blit), or isolated External
    Queue    queue = Queue::Graphics;
    u32 firstPosition = 0;
    u32 lastPosition  = 0;             // inclusive
    u32 renderGroup   = ~0u;           // index into renderGroups for Raster
};

// --- F2.3: barriers -----------------------------------------------------

enum class BarrierScope : u8 {
    Encoder, // producer and consumer in the same encoder (barrierAfterEncoderStages)
    Queue,   // producer in an earlier encoder on the same queue (barrierAfterQueueStages)
};

struct Barrier {
    BarrierScope scope          = BarrierScope::Queue;
    Stages       afterStages    = StageNone; // producer side
    Stages       beforeStages   = StageNone; // consumer side (legalised)
    bool         aliasing       = false;     // VisibilityOptionResourceAlias
    std::vector<u32> resources;              // resources that caused it (debug/dump)
};

// Barriers to encode right before the pass at `position` records its
// commands (for the first pass of a render group: at encoder start).
struct PassBarriers {
    u32 position = 0;
    std::vector<Barrier> barriers;
};

// Cross-queue synchronisation (F2.6): the producer's queue signals `value`
// on the graph's event after `signalAfterPosition`; the consumer's queue
// waits for it before `waitBeforePosition`.  Values are 1-based per frame;
// the executor offsets them by frame.
struct QueueSync {
    u32 signalAfterPosition = 0;
    u32 waitBeforePosition  = 0;
    u32 value               = 0;
};

// --- F2.2: aliasing -----------------------------------------------------

struct SizeAlign {
    u64 size  = 0;
    u64 align = 1;
};

// Heap footprint of transient resources; the Metal backend answers with
// heapTextureSizeAndAlign / heapBufferSizeAndAlign, tests with a fake.  The
// resource index lets the backend derive the usage flags from the graph's
// accesses (they change the footprint, e.g. lossless compression).
class ResourceSizer {
public:
    virtual ~ResourceSizer() = default;
    [[nodiscard]] virtual SizeAlign textureSize(u32 resource, const TextureDesc& desc) const = 0;
    [[nodiscard]] virtual SizeAlign bufferSize(u32 resource, const BufferDesc& desc) const = 0;
};

struct Placement {
    u32  resource = 0;
    u64  offset   = 0;
    u64  size     = 0;
    /// First use reuses memory of a resource that died earlier in the frame:
    /// the first pass using it needs an aliasing barrier and must not read
    /// before writing.
    bool aliased  = false;
};

struct AliasingPlan {
    std::vector<Placement> placements; // transient, non-memoryless resources
    u64 heapSize      = 0;             // bytes needed by the aliased layout
    u64 unaliasedSize = 0;             // bytes without aliasing (for the dump)
    /// OPT-1.3: max over positions of the footprints alive at that position
    /// (async resources alive all frame): the lower bound of any packing.
    u64 maxLiveSize   = 0;
};

// OPT-1.3: how transients are packed into the transient heap.
enum class AliasPolicy : u8 {
    Greedy,   // F2.2: size-descending first fit (the end-of-F4 behaviour)
    Coloring, // OPT-1.3: interval colouring, never larger than Greedy
    // OPT-1.3: Coloring where resources share memory only if the sets of
    // stage classes (raster / compute) accessing them are equal (spike 5:
    // mixing classes puts dispatch in raster first-use barriers).  May be
    // larger than Greedy.
    ColoringStageClass,
};

// OPT-1.4: which stages barriers wait for.
enum class BarrierPolicy : u8 {
    Conservative, // F2.3: first use of placed memory waits for every stage
                  // touching that memory in the frame (the end-of-F4 behaviour)
    Minimal,      // OPT-1.4: only the stages of the accesses that are not
                  // already ordered before another access of the same memory
};

// OPT-1.6: store/memoryless lint of the compiled graph.
enum class LintMode : u8 { Off, Warn, Error };

struct CompiledGraph {
    bool ok = false;
    std::vector<std::string> errors;

    /// Live passes in execution order (pass indices).  position(p) is the
    /// index of pass p in `order`, or ~0u if culled.
    std::vector<u32> order;
    std::vector<u32> positionOfPass;
    std::vector<bool> culled;          // per pass
    std::vector<Dependency> dependencies; // between live passes, deduplicated
    std::vector<Lifetime> lifetimes;   // per resource, over positions
    /// Per resource: its texture is only ever used as an attachment inside one
    /// render group and is not imported -> memoryless (no memory at all).
    std::vector<bool> memoryless;

    std::vector<RenderGroup>  renderGroups;   // F2.4
    /// Per position: index into renderGroups, or ~0u for non-raster passes.
    std::vector<u32>          groupOfPosition;
    std::vector<EncoderPlan>  encoders;       // F2.4, in execution order
    std::vector<u32>          encoderOfPosition;
    std::vector<PassBarriers> barriers;       // F2.3, sorted by position
    std::vector<QueueSync>    queueSyncs;     // F2.6
    AliasingPlan              aliasing;       // F2.2
    /// OPT-1.6: lint findings (LintMode::Warn); with LintMode::Error they
    /// are also in `errors` and compilation fails.
    std::vector<std::string>  lint;

    [[nodiscard]] u32 position(u32 pass) const {
        return pass < positionOfPass.size() ? positionOfPass[pass] : ~0u;
    }
};

struct CompileOptions {
    /// Fuse compatible adjacent raster passes (F2.4).
    bool fuseRasterPasses = true;
    /// Place transients so that non-overlapping lifetimes share memory
    /// (F2.2); needs a sizer.  Without one, no aliasing plan is produced.
    const ResourceSizer* sizer = nullptr;
    bool alias = true;
    /// OPT-1.1: execution order imposed by a plan (pass indices).  Must list
    /// every live pass exactly once and respect every dependency; otherwise
    /// compilation fails with an error naming the violation.  Empty: stable
    /// Kahn order (declaration order with the S-TBDR-6 hint).
    std::vector<u32> order;
    /// OPT-1.3 / OPT-1.4 / OPT-1.6 (defaults = end-of-F4 behaviour).
    AliasPolicy   aliasPolicy   = AliasPolicy::Greedy;
    BarrierPolicy barrierPolicy = BarrierPolicy::Conservative;
    LintMode      lint          = LintMode::Off;
};

/// Validate, cull, sort, compute lifetimes, then run the F2.4 fusion, F2.2
/// aliasing, F2.6 queue sync (and encoder split at sync points) and F2.3
/// barrier plan stages.
CompiledGraph compile(const RenderGraph& graph, const CompileOptions& options = {});

/// Stage 1 only (F2.1): validation, culling, stable Kahn order, dependencies,
/// lifetimes.  Used by compile() and by the tests of the later stages.
CompiledGraph compileOrder(const RenderGraph& graph, const std::vector<u32>& forcedOrder = {});

} // namespace phosphor::rg
