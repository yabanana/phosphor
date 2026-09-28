#pragma once
#include <cstddef>
#include <math.h>
#include <cstdint>
typedef double CFTimeInterval;
typedef const void* CFTypeRef;
typedef const struct __CFAllocator* CFAllocatorRef;
typedef const struct __CFArray* CFArrayRef;
typedef long CFIndex;
typedef struct { CFIndex version; void* retain; void* release; void* copyDescription; void* equal; } CFArrayCallBacks;
extern "C" {
extern const CFAllocatorRef kCFAllocatorDefault;
extern const CFArrayCallBacks kCFTypeArrayCallBacks;
CFArrayRef CFArrayCreate(CFAllocatorRef, const void** values, CFIndex numValues, const CFArrayCallBacks*);
}
