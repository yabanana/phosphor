#pragma once
// metal-cpp selects objc_msgSend ABI using Apple's spelling for ARM64.
// Linux clang uses __aarch64__; expose its equivalent in these syntax-only
// SDK stubs without pretending a different target architecture is available.
#if defined(__aarch64__) && !defined(__arm64__)
#define __arm64__ 1
#endif
typedef struct objc_class* Class;
typedef struct objc_object { Class isa; }* id;
typedef struct objc_selector* SEL;
typedef struct objc_object Protocol;
extern "C" {
Class objc_lookUpClass(const char* name);
Protocol* objc_getProtocol(const char* name);
SEL sel_registerName(const char* str);
}
