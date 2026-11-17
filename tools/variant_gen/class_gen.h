#pragma once

#include <string>

/// Function-constant indices the class generator writes into the plan (from the reserved constants of variants.def).
struct ClassGenConstants {
    unsigned materialFeatures = 0; // MATERIAL_FEATURES
    unsigned resolveBin = 0;       // RESOLVE_BIN
    unsigned sssOutput = 0;        // SSS_OUTPUT
};

/// `variant_gen classes`: see class_gen.cpp.  Returns the process exit code.
int generateClasses(const std::string& defPath, const std::string& outputDir, const ClassGenConstants& constants);
