#pragma once
#include "renderer/gpu_types.h"

namespace phosphor {
struct GPUShadingHistory {
    u32 generation, primitive;
    float luminance, roughness;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUShadingHistory) == 16, "adaptive shading history layout");
PHOSPHOR_GPU_CONSTANT u32 VISIBILITY_TILE = 16;
PHOSPHOR_GPU_CONSTANT u32 VISIBILITY_CLASSES = 4;
// F15.2 (ticket 21): per-class bins.  A scene with material features bins its tiles by the feature mask of the material
// (one bin per class of the scene's class table, at most VISIBILITY_FEATURE_CLASSES, = the generated active-class budget)
// plus one overflow bin for every pixel whose class is not in the table.  binning == 2 selects this mode; 1 is the legacy
// four classes (normal mapping / emission).  RESOLVE_BIN (function constant 24) tells a resolve kernel its bin:
// VISIBILITY_BIN_CLASS | mask = the pixels whose material feature mask equals `mask`, VISIBILITY_BIN_OVERFLOW = the pixels
// outside the table (that pipeline is the all-lobes one), 0 = not a feature bin.
PHOSPHOR_GPU_CONSTANT u32 VISIBILITY_FEATURE_CLASSES = 16;
PHOSPHOR_GPU_CONSTANT u32 VISIBILITY_MAX_BINS = VISIBILITY_FEATURE_CLASSES + 1; // the last bin is the overflow bin
PHOSPHOR_GPU_CONSTANT u32 VISIBILITY_BIN_CLASS = 0x1000u;
PHOSPHOR_GPU_CONSTANT u32 VISIBILITY_BIN_OVERFLOW = 0x2000u;
PHOSPHOR_GPU_CONSTANT u32 VISIBILITY_ARGS_WORDS = VISIBILITY_MAX_BINS * 3; // indirect dispatch arguments (x, y, z per bin)
struct GPUVisibilityParams {
    u32 width, height, tilesX, tilesY;
    u32 candidateCapacity, materialCount, defaultNormal, debugMode;
    float exposure, mipBias;
    u32 binning, pad; // binning: 0 off, 1 legacy classes, 2 feature classes
    u32 outputWidth, outputHeight, pad2, classCount; // classCount: binning == 2, entries of classMasks in use
    float background[3]; u32 pad4; // radiance of pixels without geometry
    u32 classMasks[16]; // VISIBILITY_FEATURE_CLASSES; binning == 2: the feature mask of each class bin, in bin order
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUVisibilityParams) == 144, "visibility parameters");
} // namespace phosphor
