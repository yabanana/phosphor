// CPU allocation tracking for Tracy (F4.2).  Compiled only with
// PHOSPHOR_TRACY: replaces the global operator new/delete so every C++
// allocation shows up in Tracy's memory view (pool "cpu").  GPU allocations
// are reported by GpuMemory under one pool per MemoryCategory.
#ifdef PHOSPHOR_TRACY

#include <tracy/Tracy.hpp>

#include <cstdlib>
#include <new>

namespace {

void* allocate(std::size_t size) {
    void* p = std::malloc(size ? size : 1);
    if (p) TracyAllocN(p, size, "cpu");
    return p;
}

void* allocateAligned(std::size_t size, std::align_val_t align) {
    void* p = nullptr;
    std::size_t a = static_cast<std::size_t>(align);
    if (a < sizeof(void*)) a = sizeof(void*);
    if (posix_memalign(&p, a, size ? size : 1) != 0) return nullptr;
    TracyAllocN(p, size, "cpu");
    return p;
}

void release(void* p) {
    if (!p) return;
    TracyFreeN(p, "cpu");
    std::free(p);
}

void* allocOrThrow(std::size_t size) {
    void* p = allocate(size);
    if (!p) throw std::bad_alloc();
    return p;
}

void* allocAlignedOrThrow(std::size_t size, std::align_val_t align) {
    void* p = allocateAligned(size, align);
    if (!p) throw std::bad_alloc();
    return p;
}

} // namespace

void* operator new(std::size_t size) { return allocOrThrow(size); }
void* operator new[](std::size_t size) { return allocOrThrow(size); }
void* operator new(std::size_t size, const std::nothrow_t&) noexcept { return allocate(size); }
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept { return allocate(size); }
void* operator new(std::size_t size, std::align_val_t a) { return allocAlignedOrThrow(size, a); }
void* operator new[](std::size_t size, std::align_val_t a) { return allocAlignedOrThrow(size, a); }
void* operator new(std::size_t size, std::align_val_t a, const std::nothrow_t&) noexcept {
    return allocateAligned(size, a);
}
void* operator new[](std::size_t size, std::align_val_t a, const std::nothrow_t&) noexcept {
    return allocateAligned(size, a);
}

void operator delete(void* p) noexcept { release(p); }
void operator delete[](void* p) noexcept { release(p); }
void operator delete(void* p, std::size_t) noexcept { release(p); }
void operator delete[](void* p, std::size_t) noexcept { release(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { release(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { release(p); }
void operator delete(void* p, std::align_val_t) noexcept { release(p); }
void operator delete[](void* p, std::align_val_t) noexcept { release(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { release(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { release(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { release(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { release(p); }

#endif // PHOSPHOR_TRACY
