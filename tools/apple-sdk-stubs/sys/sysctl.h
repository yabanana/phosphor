// Stub for Linux syntax checking (tools/apple-sdk-stubs/README.md).
#pragma once
#include <cstddef>
extern "C" int sysctlbyname(const char* name, void* oldp, size_t* oldlenp, void* newp, size_t newlen);
