#pragma once
#include <objc/runtime.h>
extern "C" {
void objc_msgSend(void);
void objc_msgSend_fpret(void);
void objc_msgSend_stret(void);
}
