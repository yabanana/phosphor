#include "pipeline/pipeline_key.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace phosphor::pipe {

namespace {

const char* formatToken(rg::Format f) {
    switch (f) {
        case rg::Format::Unknown:              return "Unknown";
        case rg::Format::R8Unorm:              return "R8Unorm";
        case rg::Format::RG8Unorm:             return "RG8Unorm";
        case rg::Format::RGBA8Unorm:           return "RGBA8Unorm";
        case rg::Format::RGBA8Srgb:            return "RGBA8Srgb";
        case rg::Format::BGRA8Unorm:           return "BGRA8Unorm";
        case rg::Format::BGRA8Srgb:            return "BGRA8Srgb";
        case rg::Format::R16Float:             return "R16Float";
        case rg::Format::RG16Float:            return "RG16Float";
        case rg::Format::RGBA16Float:          return "RGBA16Float";
        case rg::Format::R32Float:             return "R32Float";
        case rg::Format::RG32Float:            return "RG32Float";
        case rg::Format::RGBA32Float:          return "RGBA32Float";
        case rg::Format::R32Uint:              return "R32Uint";
        case rg::Format::RG11B10Float:         return "RG11B10Float";
        case rg::Format::RGB10A2Unorm:         return "RGB10A2Unorm";
        case rg::Format::Depth16Unorm:         return "Depth16Unorm";
        case rg::Format::Depth32Float:         return "Depth32Float";
        case rg::Format::Depth32FloatStencil8: return "Depth32FloatStencil8";
    }
    return "Invalid";
}

} // namespace

// Canonical format (STABLE: changing it invalidates every harvested key):
//   <kind>|<function0>|<function1>|<constants>|<outputs>
//   kind       "R" (render) or "C" (compute)
//   constants  comma separated "c<index>:<t>=<value>" in declaration order,
//              t = b (bool, 0/1) | u (uint, decimal) | i (int, signed decimal)
//                | f (float, "0x" + 8 hex digits of the bit pattern)
//   outputs    comma separated, only used attachments:
//              "o<i>=unspecialized" or "o<i>=<FormatName>/<none|over>/<mask hex>"
std::string canonicalString(const PipelineDesc& desc) {
    std::string s;
    s.reserve(96);
    s += desc.kind == PipelineKind::Compute ? 'C' : 'R';
    s += '|';
    s += desc.functions[0];
    s += '|';
    s += desc.functions[1];
    s += '|';

    char buf[32];
    for (u32 i = 0; i < desc.constantCount; ++i) {
        const FunctionConstant& c = desc.constants[i];
        if (i) s += ',';
        s += 'c';
        s += std::to_string(c.index);
        s += ':';
        switch (c.type) {
            case ConstantType::Bool:
                s += "b=";
                s += c.bits ? '1' : '0';
                break;
            case ConstantType::UInt:
                s += "u=";
                s += std::to_string(c.bits);
                break;
            case ConstantType::Int:
                s += "i=";
                s += std::to_string(static_cast<int32_t>(c.bits));
                break;
            case ConstantType::Float:
                std::snprintf(buf, sizeof(buf), "f=0x%08x", static_cast<unsigned>(c.bits));
                s += buf;
                break;
        }
    }
    s += '|';

    bool first = true;
    for (u32 i = 0; i < desc.colorCount && i < MAX_COLOR_ATTACHMENTS; ++i) {
        const ColorOutput& o = desc.color[i];
        if (!o.unspecialized && o.format == rg::Format::Unknown) continue;
        if (!first) s += ',';
        first = false;
        s += 'o';
        s += std::to_string(i);
        s += '=';
        if (o.unspecialized) {
            s += "unspecialized";
        } else {
            s += formatToken(o.format);
            s += '/';
            s += o.blend == ColorOutput::Blend::AlphaOver ? "over" : "none";
            s += '/';
            std::snprintf(buf, sizeof(buf), "%X", static_cast<unsigned>(o.writeMask & 0xF));
            s += buf;
        }
    }
    return s;
}

u64 hashBytes(const void* data, size_t size, u64 seed) {
    // FNV-1a 64 (seed xored into the offset basis), then a splitmix64 finalizer.
    u64 h = 0xcbf29ce484222325ull ^ seed;
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ull;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebull;
    h ^= h >> 31;
    return h;
}

PipelineKey pipelineKey(const PipelineDesc& desc, u32 salt) {
    const std::string s = canonicalString(desc);
    // Seed 0 is the plain hash, so salt 0 and generic descriptors agree.
    return hashBytes(s.data(), s.size(), desc.isGeneric() ? 0 : static_cast<u64>(salt));
}

} // namespace phosphor::pipe
