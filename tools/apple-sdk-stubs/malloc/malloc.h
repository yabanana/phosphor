#pragma once
// Minimal stub of <malloc/malloc.h> for the Linux syntax check
// (src/app/engine.cpp samples CPU heap statistics in benchmark mode).
#include <stddef.h>
typedef struct _malloc_zone_t malloc_zone_t;
typedef struct malloc_statistics_t {
    unsigned blocks_in_use;
    size_t   size_in_use;
    size_t   max_size_in_use;
    size_t   size_allocated;
} malloc_statistics_t;
void malloc_zone_statistics(malloc_zone_t* zone, malloc_statistics_t* stats);
