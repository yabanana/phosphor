#pragma once
typedef struct dispatch_queue_s* dispatch_queue_t;
typedef struct dispatch_data_s* dispatch_data_t;

// Memory-pressure dispatch source (src/platform/metal/memory_pressure.cpp).
#include <stdint.h>
typedef struct dispatch_source_s* dispatch_source_t;
typedef const struct dispatch_source_type_s* dispatch_source_type_t;
typedef void (*dispatch_function_t)(void*);
typedef void* dispatch_object_t;
extern const struct dispatch_source_type_s _dispatch_source_type_memorypressure;
#define DISPATCH_SOURCE_TYPE_MEMORYPRESSURE (&_dispatch_source_type_memorypressure)
#define DISPATCH_MEMORYPRESSURE_NORMAL   0x01
#define DISPATCH_MEMORYPRESSURE_WARN     0x02
#define DISPATCH_MEMORYPRESSURE_CRITICAL 0x04
dispatch_queue_t dispatch_get_main_queue(void);
dispatch_source_t dispatch_source_create(dispatch_source_type_t type, uintptr_t handle, uintptr_t mask,
                                         dispatch_queue_t queue);
void dispatch_source_set_event_handler_f(dispatch_source_t source, dispatch_function_t handler);
uintptr_t dispatch_source_get_data(dispatch_source_t source);
void dispatch_source_cancel(dispatch_source_t source);
void dispatch_set_context(dispatch_object_t object, void* context);
void* dispatch_get_context(dispatch_object_t object);
void dispatch_resume(dispatch_object_t object);
void dispatch_release(dispatch_object_t object);
