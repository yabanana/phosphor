#pragma once
typedef struct objc_class* Class;
typedef struct objc_object { Class isa; }* id;
typedef struct objc_selector* SEL;
typedef struct objc_object Protocol;
extern "C" {
Class objc_lookUpClass(const char* name);
Protocol* objc_getProtocol(const char* name);
SEL sel_registerName(const char* str);
}
