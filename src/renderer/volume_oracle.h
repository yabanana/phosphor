#pragma once
#include "renderer/atmosphere.h"
#include <array>
#include <span>
#include <string>
#include <vector>
namespace phosphor {
struct VolumeOracleSettings {
    // Frozen before execution; exploratory sparse gates, not certification.
    double transAbsolute=0.02,transRelative=0.05;
    double multipleAbsolute=0.02,multipleRelative=0.25;
    double skyAbsolute=0.02,skyRelative=0.25;
    double fogAbsolute=2e-5,fogRelative=2e-4;
    double solarAbsolute=0.01,solarRelative=2e-4;
};
struct VolumeOracleCase {
    std::string kind;
    u32 x=0,y=0,index=0,expectedEpoch=0,actualEpoch=0;
    std::array<double,4> expected{},actual{};
    double absoluteError=0,relativeError=0,absoluteTolerance=0,relativeTolerance=0;
    bool passed=false;
};
struct VolumeOracleInput {
    GPUAtmosphereParams atmosphere{};
    GPUFogParams fog{};
    GPUVolumeDiagnosticParams diagnostics{};
    bool homogeneousFog=false;
};
// CPU references: adaptive-Simpson optical depth/radiance and independent
// Gauss-Legendre angular closure, with CPU-only memoized multiscattering nodes.
std::vector<VolumeOracleCase> evaluateVolumeOracle(const VolumeOracleInput&,
    std::span<const GPUVolumeNumericSample>,const VolumeOracleSettings& settings={});
} // namespace phosphor
