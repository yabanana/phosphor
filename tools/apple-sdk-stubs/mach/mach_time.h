#pragma once
// Minimal stub of <mach/mach_time.h> for the Linux syntax check
// (metal_syntax_check); never linked.
#include <stdint.h>

typedef struct mach_timebase_info {
    uint32_t numer;
    uint32_t denom;
} mach_timebase_info_data_t;

uint64_t mach_absolute_time(void);
int mach_timebase_info(mach_timebase_info_data_t* info);
