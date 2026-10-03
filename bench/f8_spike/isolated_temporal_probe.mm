// F8.4 spike: one unchanged MetalFX temporal scaler per disposable process.
// Shared file mapping and a private socketpair; no network service or private API.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <libproc.h>
#include <sys/resource.h>
extern char **environ;

struct Layout {
    uint64_t offset[6], row[6], bytes;
    uint32_t width, height;
};
static uint64_t alignUp(uint64_t n, uint64_t a) {
    return (n + a - 1) / a * a;
}
static Layout layout(uint32_t w, uint32_t h) {
    Layout l{};
    l.width = w;
    l.height = h;
    const unsigned bpp[] = {8, 4, 4, 1, 2, 8};
    for (unsigned i = 0; i < 6; ++i) {
        l.offset[i] = l.bytes;
        l.row[i] = alignUp((i == 4 ? 1 : w) * bpp[i], 256);
        l.bytes += l.row[i] * (i == 4 ? 1 : h);
    }
    l.bytes = alignUp(l.bytes, getpagesize());
    return l;
}
struct Frame {
    uint32_t command = 1, width = 0, height = 0, reset = 1;
    float jitterX = 0, jitterY = 0;
};
struct Reply {
    uint64_t allocated = 0, footprint = 0;
    int32_t status = 0;
};
static uint64_t footprint() {
    rusage_info_v4 r{};
    return proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&r) == 0 ? r.ri_phys_footprint : 0;
}
static bool transfer(int fd, void *data, size_t size, bool send, int timeout = 30000) {
    auto *p = static_cast<char *>(data);
    while (size) {
        pollfd f{fd, short(send ? POLLOUT : POLLIN), 0};
        int r = poll(&f, 1, timeout);
        if (r <= 0)
            return false;
        ssize_t n = send ? write(fd, p, size) : read(fd, p, size);
        if (n <= 0)
            return false;
        p += n;
        size -= size_t(n);
    }
    return true;
}
static int worker() {
    Layout l{};
    if (!transfer(3, &l, sizeof(l), false))
        return 2;
    if (l.width < 32 || l.width > 1920 || l.height < 32 || l.height > 1080 ||
        std::memcmp(&l, &(const Layout &)(layout(l.width, l.height)), sizeof(l)))
        return 2;
    void *memory = mmap(nullptr, l.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, 4, 0);
    if (memory == MAP_FAILED)
        return 2;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        auto *cd = [MTL4CompilerDescriptor new];
        id<MTL4Compiler> compiler = [device newCompilerWithDescriptor:cd error:nil];
        [cd release];
        auto *d = [MTLFXTemporalScalerDescriptor new];
        d.inputWidth = d.outputWidth = l.width;
        d.inputHeight = d.outputHeight = l.height;
        d.colorTextureFormat = d.outputTextureFormat = MTLPixelFormatRGBA16Float;
        d.depthTextureFormat = MTLPixelFormatDepth32Float;
        d.motionTextureFormat = MTLPixelFormatRG16Float;
        d.reactiveMaskTextureEnabled = YES;
        d.reactiveMaskTextureFormat = MTLPixelFormatR8Unorm;
        d.requiresSynchronousInitialization = YES;
        d.inputContentPropertiesEnabled = YES;
        d.inputContentMinScale = 1;
        d.inputContentMaxScale = 2;
        id<MTL4FXTemporalScaler> scaler = [d newTemporalScalerWithDevice:device compiler:compiler];
        [d release];
        if (!scaler)
            return 77;
        id<MTLBuffer> bridge = [device newBufferWithBytesNoCopy:memory
                                                         length:l.bytes
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        const MTLPixelFormat formats[] = {MTLPixelFormatRGBA16Float, MTLPixelFormatDepth32Float,
                                          MTLPixelFormatRG16Float,   MTLPixelFormatR8Unorm,
                                          MTLPixelFormatR16Float,    MTLPixelFormatRGBA16Float};
        id<MTLTexture> t[6];
        auto *rd = [MTLResidencySetDescriptor new];
        id<MTLResidencySet> rs = [device newResidencySetWithDescriptor:rd error:nil];
        [rd release];
        [rs addAllocation:bridge];
        for (unsigned i = 0; i < 6; ++i) {
            auto *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:formats[i]
                                                                          width:i == 4 ? 1 : l.width
                                                                         height:i == 4 ? 1 : l.height
                                                                      mipmapped:NO];
            td.storageMode = MTLStorageModePrivate;
            td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
            if (i != 1)
                td.usage |= MTLTextureUsageShaderWrite;
            t[i] = [device newTextureWithDescriptor:td];
            [rs addAllocation:t[i]];
        }
        id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
        [rs commit];
        [queue addResidencySet:rs];
        id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
        id<MTL4CommandBuffer> cb = [device newCommandBuffer];
        id<MTLSharedEvent> done = [device newSharedEvent];
        id<MTLFence> fence = [device newFence];
        uint64_t sequence = 0;
        Reply reply{device.currentAllocatedSize, footprint(), 0};
        if (!transfer(3, &reply, sizeof(reply), true))
            return 2;
        for (;;) {
            Frame frame{};
            if (!transfer(3, &frame, sizeof(frame), false))
                break;
            if (!frame.command)
                break;
            if (frame.width < 32 || frame.width > l.width || frame.height < 32 || frame.height > l.height ||
                2 * frame.width < l.width || 2 * frame.height < l.height)
                return 2;
            @autoreleasepool {
                [allocator reset];
                [cb beginCommandBufferWithAllocator:allocator];
                id<MTL4ComputeCommandEncoder> copy = [cb computeCommandEncoder];
                for (unsigned i = 0; i < 5; ++i)
                    [copy copyFromBuffer:bridge
                               sourceOffset:l.offset[i]
                          sourceBytesPerRow:l.row[i]
                        sourceBytesPerImage:l.row[i] * (i == 4 ? 1 : l.height)
                                 sourceSize:MTLSizeMake(i == 4 ? 1 : l.width, i == 4 ? 1 : l.height, 1)
                                  toTexture:t[i]
                           destinationSlice:0
                           destinationLevel:0
                          destinationOrigin:MTLOriginMake(0, 0, 0)];
                [copy updateFence:fence afterEncoderStages:MTLStageBlit];
                [copy endEncoding];
                scaler.colorTexture = t[0];
                scaler.depthTexture = t[1];
                scaler.motionTexture = t[2];
                scaler.reactiveMaskTexture = t[3];
                scaler.exposureTexture = t[4];
                scaler.outputTexture = t[5];
                scaler.inputContentWidth = frame.width;
                scaler.inputContentHeight = frame.height;
                scaler.motionVectorScaleX = scaler.motionVectorScaleY = 1;
                scaler.jitterOffsetX = frame.jitterX;
                scaler.jitterOffsetY = frame.jitterY;
                scaler.depthReversed = YES;
                scaler.reset = frame.reset;
                scaler.preExposure = 1;
                scaler.fence = fence;
                [scaler encodeToCommandBuffer:cb];
                copy = [cb computeCommandEncoder];
                [copy waitForFence:fence beforeEncoderStages:MTLStageBlit];
                [copy copyFromTexture:t[5]
                                 sourceSlice:0
                                 sourceLevel:0
                                sourceOrigin:MTLOriginMake(0, 0, 0)
                                  sourceSize:MTLSizeMake(l.width, l.height, 1)
                                    toBuffer:bridge
                           destinationOffset:l.offset[5]
                      destinationBytesPerRow:l.row[5]
                    destinationBytesPerImage:l.row[5] * l.height];
                [copy endEncoding];
                [cb endCommandBuffer];
                id<MTL4CommandBuffer> batch[] = {cb};
                [queue commit:batch count:1];
                [queue signalEvent:done value:++sequence];
                reply.status = [done waitUntilSignaledValue:sequence timeoutMS:1000] ? 0 : 1;
                reply.allocated = device.currentAllocatedSize;
                reply.footprint = footprint();
                if (!transfer(3, &reply, sizeof(reply), true, 1000) || reply.status)
                    return 2;
            }
        }
        [queue removeResidencySet:rs];
        [cb release];
        [allocator release];
        [done release];
        [fence release];
        [rs release];
        [queue release];
        [scaler release];
        for (auto texture : t)
            [texture release];
        [bridge release];
        [compiler release];
        [device release];
        // MetalFX's cycle is still present inside this process. The lifecycle
        // boundary being tested is the OS process, never an extra release.
    }
    munmap(memory, l.bytes);
    close(4);
    close(3);
    return 0;
}
int main(int argc, char **argv) {
    if (argc == 2 && std::strcmp(argv[1], "--worker") == 0)
        return worker();
    const int cycles = argc > 1 ? std::atoi(argv[1]) : 8;
    if (cycles < 1 || cycles > 32)
        return 2;
    signal(SIGPIPE, SIG_IGN);
    for (int cycle = 0; cycle < cycles; ++cycle) {
        Layout l = layout(cycle % 2 ? 640 : 1280, cycle % 2 ? 360 : 720);
        char name[] = "/tmp/phosphor-metalfx-XXXXXX";
        int fd = mkstemp(name);
        if (fd < 0)
            return 2;
        unlink(name);
        if (ftruncate(fd, l.bytes))
            return 2;
        void *memory = mmap(nullptr, l.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (memory == MAP_FAILED)
            return 2;
        // Put inherited descriptors above fixed child slots before dup2.
        int inherited = fcntl(fd, F_DUPFD_CLOEXEC, 10);
        close(fd);
        fd = inherited;
        int channel[2];
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, channel))
            return 2;
        int childSocket = fcntl(channel[1], F_DUPFD_CLOEXEC, 10);
        close(channel[1]);
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, childSocket, 3);
        posix_spawn_file_actions_adddup2(&actions, fd, 4);
        posix_spawn_file_actions_addclose(&actions, childSocket);
        posix_spawn_file_actions_addclose(&actions, fd);
        if (channel[0] != 3 && channel[0] != 4)
            posix_spawn_file_actions_addclose(&actions, channel[0]);
        char *args[] = {argv[0], const_cast<char *>("--worker"), nullptr};
        pid_t child = 0;
        int rc = posix_spawn(&child, argv[0], &actions, nullptr, args, environ);
        posix_spawn_file_actions_destroy(&actions);
        close(childSocket);
        if (rc)
            return 2;
        if (!transfer(channel[0], &l, sizeof(l), true))
            return 2;
        Reply reply{};
        if (!transfer(channel[0], &reply, sizeof(reply), false) || reply.status)
            return 2;
        uint64_t maxAllocated = reply.allocated, maxFootprint = reply.footprint;
        for (int f = 0; f < 12; ++f) {
            std::memset(memory, 0, l.bytes);
            for (unsigned y = 0; y < l.height; ++y)
                for (unsigned x = 0; x < l.width; ++x) {
                    auto *color = reinterpret_cast<_Float16 *>((char *)memory + l.offset[0] + y * l.row[0]) + 4 * x;
                    color[0] = .25f;
                    color[1] = .5f;
                    color[2] = .75f;
                    color[3] = 1;
                    auto *depth = reinterpret_cast<float *>((char *)memory + l.offset[1] + y * l.row[1]);
                    depth[x] = .5f;
                }
            *reinterpret_cast<_Float16 *>((char *)memory + l.offset[4]) = 1;
            const bool reduced = f % 4 >= 2;
            Frame frame{1, l.width / (reduced ? 2 : 1), l.height / (reduced ? 2 : 1), uint32_t(f % 2 == 0)};
            if (!transfer(channel[0], &frame, sizeof(frame), true, 1000) ||
                !transfer(channel[0], &reply, sizeof(reply), false, 1000) || reply.status)
                return 2;
            maxAllocated = std::max(maxAllocated, reply.allocated);
            maxFootprint = std::max(maxFootprint, reply.footprint);
            auto *pixel = reinterpret_cast<_Float16 *>((char *)memory + l.offset[5] + (l.height / 2) * l.row[5]) +
                          4 * (l.width / 2);
            for (int c = 0; c < 3; ++c)
                if (!std::isfinite(float(pixel[c])) || std::abs(float(pixel[c]) - .25f * (c + 1)) > .06f) {
                    printf("OUTPUT FAIL %f\n", float(pixel[c]));
                    return 1;
                }
        }
        Frame stop{};
        stop.command = 0;
        transfer(channel[0], &stop, sizeof(stop), true);
        close(channel[0]);
        int status = 0;
        if (waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
            return 2;
        munmap(memory, l.bytes);
        close(fd);
        printf("CYCLE %d pid=%d reaped=1 size=%ux%u child_device_peak=%llu child_footprint_peak=%llu "
               "parent_footprint=%llu\n",
               cycle, child, l.width, l.height, maxAllocated, maxFootprint, footprint());
        fflush(stdout);
    }
    printf("ISOLATED TEMPORAL PROBE PASS\n");
    return 0;
}
