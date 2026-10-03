#include <MetalFX/MetalFX.hpp>
#include "platform/metal/temporal_worker.h"
#include "platform/metal/metal_context.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/pipeline_cache.h"
#include "core/log.h"
#include <SDL3/SDL.h>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <libproc.h>
#include <mach-o/dyld.h>
#include <pthread/qos.h>
#endif
extern char **environ;

namespace phosphor {
namespace {
using namespace temporal_worker;
constexpr int FrameTimeoutMs = 1000;
std::atomic<u64> spawned{0}, reaped{0}, peakLive{0}, failures{0}, mapped{0}, liveWorkers{0}, workerSlots{0},
    workerGpuAllocations{0};
constexpr u64 MaxLiveWorkers = 8; // Four views plus one retiring generation.
NS::String *str(const char *s) {
    return NS::String::string(s, NS::UTF8StringEncoding);
}
bool transfer(int fd, void *data, size_t size, bool send, int timeoutMs) {
    auto *p = static_cast<char *>(data);
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (size) {
        const auto left =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - std::chrono::steady_clock::now()).count();
        if (left <= 0)
            return false;
        pollfd item{fd, short(send ? POLLOUT : POLLIN), 0};
        const int result = poll(&item, 1, static_cast<int>(left));
        if (result < 0 && errno == EINTR)
            continue;
        if (result <= 0)
            return false;
        const ssize_t n = send ? write(fd, p, size) : read(fd, p, size);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            return false;
        p += n;
        size -= static_cast<size_t>(n);
    }
    return true;
}
u64 processFootprint() {
#if defined(__APPLE__)
    rusage_info_v4 r{};
    return proc_pid_rusage(getpid(), RUSAGE_INFO_V4, reinterpret_cast<rusage_info_t *>(&r)) == 0 ? r.ri_phys_footprint
                                                                                                 : 0;
#else
    return 0;
#endif
}
std::string executablePath() {
#if defined(__APPLE__)
    u32 length = 0;
    _NSGetExecutablePath(nullptr, &length);
    std::string result(length, '\0');
    if (_NSGetExecutablePath(result.data(), &length))
        throw std::runtime_error("Cannot resolve MetalFX worker executable");
    result.resize(std::strlen(result.c_str()));
    return std::filesystem::canonical(result).string();
#else
    throw std::runtime_error("MetalFX worker requires macOS");
#endif
}
void noSigpipe(int fd) {
#if defined(__APPLE__)
    const int value = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &value, sizeof(value));
#else
    (void)fd;
#endif
}
void framePriority() {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
}
} // namespace

struct TemporalWorker::Impl {
    temporal_worker::Layout layout;
    int mapFd = -1, socket = -1;
    void *mapping = MAP_FAILED;
    pid_t pid = -1;
    bool ownsSlot = false;
    MTL::SharedEvent *input = nullptr, *output = nullptr;
    std::thread thread;
    mutable std::mutex mutex;
    std::condition_variable cv;
    struct Job {
        temporal_worker::Request request;
        bool submitted = false;
    };
    std::array<Job, temporal_worker::Slots> queue{};
    u32 head = 0, count = 0;
    u64 ticket = 0;
    std::atomic<u32> stage{0};
    std::atomic<bool> ready{false}, cancelled{false}, retiring{false}, finished{false}, failed{false};
    std::atomic<u64> bytes{0}, footprint{0}, allocations{0};

    void fail(const char *reason) {
        if (!failed.exchange(true)) {
            ++failures;
            LOG_ERROR("MetalFX worker: %s", reason);
        }
    }
    void cleanup() {
        if (socket >= 0) {
            temporal_worker::Request stop;
            stop.command = temporal_worker::Command::Stop;
            transfer(socket, &stop, sizeof(stop), true, 100);
            close(socket);
            socket = -1;
        }
        if (pid > 0) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(10000);
            int status = 0;
            bool done = false;
            while (std::chrono::steady_clock::now() < deadline) {
                const auto result = waitpid(pid, &status, WNOHANG);
                if (result == pid) {
                    done = true;
                    break;
                }
                if (result < 0 && errno != EINTR)
                    break;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            if (!done) {
                kill(pid, SIGKILL);
                while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
                }
                fail("forced termination after shutdown timeout");
            } else if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
                fail("child exited unsuccessfully");
            ++reaped;
            --liveWorkers;
            pid = -1;
        }
        if (ownsSlot) {
            --workerSlots;
            ownsSlot = false;
        }
    }
    void loop() {
        framePriority();
        for (;;) {
            temporal_worker::Request request;
            {
                std::unique_lock lock(mutex);
                cv.wait(lock, [this] { return retiring.load() || (count && queue[head].submitted); });
                if (retiring && (!count || !queue[head].submitted))
                    break;
                request = queue[head].request;
                head = (head + 1) % queue.size();
                --count;
            }
            auto *pool = NS::AutoreleasePool::alloc()->init();
            // The renderer has already scheduled a GPU queue signal. Only
            // this broker waits on the CPU; the render thread keeps recording.
            bool ok = input->waitUntilSignaledValue(request.ticket, FrameTimeoutMs);
            if (!ok)
                fail("input-ready event timeout");
            temporal_worker::Reply reply;
            if (ok && !failed) {
                ok = transfer(socket, &request, sizeof(request), true, FrameTimeoutMs) &&
                     transfer(socket, &reply, sizeof(reply), false, FrameTimeoutMs) &&
                     reply.magic == temporal_worker::Magic && reply.version == temporal_worker::Version &&
                     reply.ticket == request.ticket && reply.status == 0;
                if (!ok)
                    fail("frame IPC, worker exit or GPU failure");
            } else
                ok = false;
            if (!ok) {
                // Completion is unknown: never race a possible late GPU write
                // by reusing or clearing the shared output. Match the backend's
                // GPU-timeout policy: destroy this process's queued waits.
                if (pid > 0)
                    kill(pid, SIGKILL);
                std::fflush(nullptr);
                std::_Exit(EXIT_FAILURE);
            }
            bytes = reply.deviceBytes;
            footprint = reply.physicalFootprint;
            workerGpuAllocations.fetch_add(reply.gpuAllocations - allocations.load());
            allocations = reply.gpuAllocations;
            output->setSignaledValue(request.ticket);
            pool->release();
        }
        cleanup();
        finished = true;
    }
};

TemporalWorker::TemporalWorker(MTL::Device *device, u32 width, u32 height) : impl_(std::make_unique<Impl>()) {
    auto &s = *impl_;
    s.layout = temporal_worker::makeLayout(width, height);
    char name[] = "/tmp/phosphor-metalfx-XXXXXX";
    int fd = mkstemp(name);
    if (fd < 0)
        throw std::runtime_error("Cannot create MetalFX shared mapping");
    unlink(name);
    s.mapFd = fcntl(fd, F_DUPFD_CLOEXEC, 10);
    close(fd);
    if (s.mapFd < 0 || ftruncate(s.mapFd, static_cast<off_t>(s.layout.mappedBytes))) {
        if (s.mapFd >= 0)
            close(s.mapFd);
        throw std::runtime_error("Cannot size MetalFX shared mapping");
    }
    s.mapping = mmap(nullptr, s.layout.mappedBytes, PROT_READ | PROT_WRITE, MAP_SHARED, s.mapFd, 0);
    if (s.mapping == MAP_FAILED) {
        close(s.mapFd);
        throw std::runtime_error("Cannot map MetalFX transfer memory");
    }
    s.input = device->newSharedEvent();
    s.output = device->newSharedEvent();
    if (!s.input || !s.output) {
        if (s.input)
            s.input->release();
        if (s.output)
            s.output->release();
        munmap(s.mapping, s.layout.mappedBytes);
        close(s.mapFd);
        throw std::runtime_error("Cannot create MetalFX transfer events");
    }
    s.input->setLabel(str("MetalFX worker input ready"));
    s.output->setLabel(str("MetalFX worker output ready"));
    mapped.fetch_add(s.layout.mappedBytes);
}
TemporalWorker::~TemporalWorker() {
    retire();
    if (impl_->thread.joinable())
        impl_->thread.join();
    else
        impl_->cleanup();
    impl_->input->release();
    impl_->output->release();
    munmap(impl_->mapping, impl_->layout.mappedBytes);
    close(impl_->mapFd);
    mapped.fetch_sub(impl_->layout.mappedBytes);
}
void TemporalWorker::start() {
    auto &s = *impl_;
    u32 expected = 0;
    if (!s.stage.compare_exchange_strong(expected, 1))
        return;
    try {
        if (s.cancelled || s.retiring) {
            s.finished = true;
            s.stage = 3;
            return;
        }
        u64 occupied = workerSlots.load();
        while (occupied < MaxLiveWorkers && !workerSlots.compare_exchange_weak(occupied, occupied + 1)) {
        }
        if (occupied >= MaxLiveWorkers) {
            s.stage = 0;
            if (s.retiring || s.cancelled) {
                u32 idle = 0;
                if (s.stage.compare_exchange_strong(idle, 3))
                    s.finished = true;
            }
            return;
        } // Retry later without blocking the renderer/compiler queue.
        s.ownsSlot = true;
        int channel[2];
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, channel))
            throw std::runtime_error("Cannot create MetalFX worker channel");
        noSigpipe(channel[0]);
        noSigpipe(channel[1]);
        fcntl(channel[0], F_SETFD, FD_CLOEXEC);
        fcntl(channel[1], F_SETFD, FD_CLOEXEC);
        int inherited = fcntl(channel[1], F_DUPFD_CLOEXEC, 10);
        close(channel[1]);
        if (inherited < 0) {
            close(channel[0]);
            throw std::runtime_error("Cannot inherit MetalFX worker socket");
        }
        s.socket = channel[0];
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, inherited, 3);
        posix_spawn_file_actions_adddup2(&actions, s.mapFd, 4);
        posix_spawn_file_actions_addclose(&actions, inherited);
        posix_spawn_file_actions_addclose(&actions, s.mapFd);
        if (s.socket != 3 && s.socket != 4)
            posix_spawn_file_actions_addclose(&actions, s.socket);
        std::string executable = executablePath();
        char *args[] = {executable.data(), const_cast<char *>("--metalfx-worker"), nullptr};
        posix_spawnattr_t attributes;
        posix_spawnattr_init(&attributes);
#if defined(__APPLE__)
        posix_spawnattr_setflags(&attributes, POSIX_SPAWN_CLOEXEC_DEFAULT);
        for (int fd = 0; fd < 3; ++fd)
            if (fcntl(fd, F_GETFD) >= 0)
                posix_spawn_file_actions_addinherit_np(&actions, fd);
#endif
        const int result = posix_spawn(&s.pid, executable.c_str(), &actions, &attributes, args, environ);
        posix_spawnattr_destroy(&attributes);
        posix_spawn_file_actions_destroy(&actions);
        close(inherited);
        if (result) {
            s.pid = -1;
            s.cleanup();
            throw std::runtime_error("Cannot spawn MetalFX worker");
        }
        ++spawned;
        const u64 live = liveWorkers.fetch_add(1) + 1;
        u64 old = peakLive.load();
        while (old < live && !peakLive.compare_exchange_weak(old, live)) {
        }
        temporal_worker::Reply reply;
        if (!transfer(s.socket, &s.layout, sizeof(s.layout), true, 30000) ||
            !transfer(s.socket, &reply, sizeof(reply), false, 30000) || reply.magic != temporal_worker::Magic ||
            reply.version != temporal_worker::Version || reply.status || reply.ticket) {
            s.fail("startup handshake failed");
            s.cleanup();
            throw std::runtime_error("MetalFX worker initialization failed");
        }
        s.bytes = reply.deviceBytes;
        s.footprint = reply.physicalFootprint;
        workerGpuAllocations.fetch_add(reply.gpuAllocations - s.allocations.load());
        s.allocations = reply.gpuAllocations;
        if (s.cancelled || s.retiring) {
            s.cleanup();
            s.finished = true;
            s.stage = 3;
            return;
        }
        s.thread = std::thread([&s] { s.loop(); });
        s.ready = true;
        s.stage = 2;
    } catch (...) {
        s.cleanup();
        s.finished = true;
        s.stage = 3;
        throw;
    }
}
void TemporalWorker::cancelPending() {
    if (!ready())
        impl_->cancelled = true;
}
void TemporalWorker::retire() {
    {
        std::lock_guard lock(impl_->mutex);
        impl_->retiring = true;
        u32 expected = 0;
        if (impl_->stage.compare_exchange_strong(expected, 3))
            impl_->finished = true;
    }
    impl_->cv.notify_one();
}
bool TemporalWorker::ready() const {
    return impl_->ready.load() && !impl_->cancelled && !impl_->retiring;
}
bool TemporalWorker::finished() const {
    return impl_->finished.load();
}
bool TemporalWorker::failed() const {
    return impl_->failed.load();
}
u64 TemporalWorker::enqueue(temporal_worker::Request request) {
    std::lock_guard lock(impl_->mutex);
    if (!ready() || impl_->failed || impl_->count == impl_->queue.size())
        throw std::runtime_error("MetalFX worker unavailable or queue overflow");
    request.ticket = ++impl_->ticket;
    const bool crash = request.command == temporal_worker::Command::CrashForTest;
    if (crash)
        request.command = temporal_worker::Command::Frame;
    if (!temporal_worker::valid(request, impl_->layout))
        throw std::invalid_argument("Invalid MetalFX worker frame");
    if (crash)
        request.command = temporal_worker::Command::CrashForTest;
    impl_->queue[(impl_->head + impl_->count) % impl_->queue.size()] = {request, false};
    ++impl_->count;
    impl_->cv.notify_one();
    return request.ticket;
}
void TemporalWorker::submitted(u64 ticket) {
    {
        std::lock_guard lock(impl_->mutex);
        for (u32 i = 0; i < impl_->count; ++i) {
            auto &job = impl_->queue[(impl_->head + i) % impl_->queue.size()];
            if (job.request.ticket == ticket && !job.submitted) {
                job.submitted = true;
                impl_->cv.notify_one();
                return;
            }
        }
    }
    impl_->fail("invalid submission publication");
    std::fflush(nullptr);
    std::_Exit(EXIT_FAILURE);
}
MTL::SharedEvent *TemporalWorker::inputReady() const {
    return impl_->input;
}
MTL::SharedEvent *TemporalWorker::outputReady() const {
    return impl_->output;
}
void *TemporalWorker::mapping() const {
    return impl_->mapping;
}
const temporal_worker::Layout &TemporalWorker::layout() const {
    return impl_->layout;
}
u64 TemporalWorker::deviceBytes() const {
    return impl_->finished ? 0 : impl_->bytes.load();
}
u64 TemporalWorker::physicalFootprint() const {
    return impl_->finished ? 0 : impl_->footprint.load();
}
u64 TemporalWorker::gpuAllocations() const {
    return impl_->allocations.load();
}
u64 TemporalWorker::spawnedCount() {
    return spawned.load();
}
u64 TemporalWorker::reapedCount() {
    return reaped.load();
}
u64 TemporalWorker::peakLiveCount() {
    return peakLive.load();
}
u64 TemporalWorker::failureCount() {
    return failures.load();
}
u64 TemporalWorker::totalGpuAllocations() {
    return workerGpuAllocations.load();
}
u64 TemporalWorker::mappedBytes() {
    return mapped.load();
}
u64 TemporalWorker::localPhysicalFootprint() {
    return processFootprint();
}

int runTemporalWorker() {
    using namespace temporal_worker;
    noSigpipe(3);
    framePriority();
    Layout layout;
    struct stat file{};
    if (!transfer(3, &layout, sizeof(layout), false, 30000) || !valid(layout) || fstat(4, &file) || file.st_size < 0 ||
        u64(file.st_size) != layout.mappedBytes)
        return 2;
    void *mapping = mmap(nullptr, layout.mappedBytes, PROT_READ | PROT_WRITE, MAP_SHARED, 4, 0);
    if (mapping == MAP_FAILED)
        return 2;
    close(4);
    const auto backing = std::shared_ptr<void>(mapping, [bytes = layout.mappedBytes](void *p) { munmap(p, bytes); });
    int exitCode = 0;
    try {
        const char *basePath = SDL_GetBasePath();
        if (!basePath)
            throw std::runtime_error("Cannot locate worker shader resources");
        const std::filesystem::path base(basePath);
        MetalContext context(nullptr, (base / "shaders/phosphor.metallib").string());
        PipelineCache::Options options;
        PipelineCache pipelines(context, options);
        auto *descriptor = MTLFX::TemporalScalerDescriptor::alloc()->init();
        descriptor->setInputWidth(layout.width);
        descriptor->setInputHeight(layout.height);
        descriptor->setOutputWidth(layout.width);
        descriptor->setOutputHeight(layout.height);
        descriptor->setColorTextureFormat(MTL::PixelFormatRGBA16Float);
        descriptor->setOutputTextureFormat(MTL::PixelFormatRGBA16Float);
        descriptor->setDepthTextureFormat(MTL::PixelFormatDepth32Float);
        descriptor->setMotionTextureFormat(MTL::PixelFormatRG16Float);
        descriptor->setReactiveMaskTextureEnabled(true);
        descriptor->setReactiveMaskTextureFormat(MTL::PixelFormatR8Unorm);
        descriptor->setAutoExposureEnabled(false);
        descriptor->setRequiresSynchronousInitialization(true);
        descriptor->setInputContentPropertiesEnabled(true);
        descriptor->setInputContentMinScale(1);
        descriptor->setInputContentMaxScale(2);
        auto pending = pipelines.requestTemporalScaler(descriptor);
        descriptor->release();
        auto scaler = pending.get();
        if (!scaler)
            throw std::runtime_error("Worker MetalFX unavailable");
        auto *bridge = context.memory().newSharedBuffer(
            mapping, layout.mappedBytes,
            ^(void *, NS::UInteger) {
              (void)backing;
            },
            MemoryCategory::Other, "MetalFX worker shared bridge");
        constexpr MTL::PixelFormat formats[] = {MTL::PixelFormatRGBA16Float, MTL::PixelFormatDepth32Float,
                                                MTL::PixelFormatRG16Float,   MTL::PixelFormatR8Unorm,
                                                MTL::PixelFormatR16Float,    MTL::PixelFormatRGBA16Float};
        std::array<MTL::Texture *, Planes> textures{};
        for (u32 i = 0; i < Planes; ++i) {
            auto *d = MTL::TextureDescriptor::texture2DDescriptor(formats[i], i == Exposure ? 1 : layout.width,
                                                                  i == Exposure ? 1 : layout.height, false);
            d->setStorageMode(MTL::StorageModePrivate);
            d->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget |
                        (i == Depth ? MTL::TextureUsageUnknown : MTL::TextureUsageShaderWrite));
            textures[i] = context.memory().newTexture(d, MemoryCategory::RenderTargets, "MetalFX worker texture");
            if (!textures[i])
                throw std::runtime_error("Worker texture allocation failed");
        }
        context.commitResidency();
        auto *allocator = context.device()->newCommandAllocator();
        auto *command = context.device()->newCommandBuffer();
        auto *done = context.device()->newSharedEvent();
        auto *fence = context.device()->newFence();
        Reply reply;
        reply.deviceBytes = context.device()->currentAllocatedSize();
        reply.physicalFootprint = processFootprint();
        reply.gpuAllocations = context.memory().allocationCount();
        if (!transfer(3, &reply, sizeof(reply), true, 30000))
            throw std::runtime_error("Worker handshake write failed");
        u64 previous = 0;
        for (;;) {
            Request request;
            pollfd incoming{3, POLLIN, 0};
            int available;
            do {
                available = poll(&incoming, 1, -1);
            } while (available < 0 && errno == EINTR);
            if (available <= 0 || !transfer(3, &request, sizeof(request), false, FrameTimeoutMs))
                break;
            if (request.magic != Magic || request.version != Version)
                throw std::runtime_error("Worker protocol mismatch");
            if (request.command == Command::Stop)
                break;
            if (request.command == Command::CrashForTest)
                std::_Exit(86);
            if (!valid(request, layout) || request.ticket <= previous)
                throw std::runtime_error("Invalid worker input");
            if (request.delayMs)
                std::this_thread::sleep_for(std::chrono::milliseconds(request.delayMs));
            auto *pool = NS::AutoreleasePool::alloc()->init();
            allocator->reset();
            command->beginCommandBuffer(allocator);
            auto *copy = command->computeCommandEncoder();
            const u64 slotBase = request.slot * layout.slotBytes;
            for (u32 i = 0; i < Output; ++i)
                copy->copyFromBuffer(
                    bridge, slotBase + layout.offsets[i], layout.rows[i],
                    layout.rows[i] * (i == Exposure ? 1 : layout.height),
                    MTL::Size::Make(i == Exposure ? 1 : layout.width, i == Exposure ? 1 : layout.height, 1),
                    textures[i], 0, 0, MTL::Origin::Make(0, 0, 0));
            copy->updateFence(fence, MTL::StageBlit);
            copy->endEncoding();
            scaler->setColorTexture(textures[Color]);
            scaler->setDepthTexture(textures[Depth]);
            scaler->setMotionTexture(textures[Motion]);
            scaler->setReactiveMaskTexture(textures[Reactive]);
            scaler->setExposureTexture(textures[Exposure]);
            scaler->setOutputTexture(textures[Output]);
            scaler->setInputContentWidth(request.inputWidth);
            scaler->setInputContentHeight(request.inputHeight);
            scaler->setMotionVectorScaleX(request.motionScale);
            scaler->setMotionVectorScaleY(request.motionScale);
            scaler->setJitterOffsetX(request.jitterX);
            scaler->setJitterOffsetY(request.jitterY);
            scaler->setReset(request.reset);
            scaler->setDepthReversed(true);
            scaler->setPreExposure(1);
            scaler->setFence(fence);
            scaler->encodeToCommandBuffer(command);
            copy = command->computeCommandEncoder();
            copy->waitForFence(fence, MTL::StageBlit);
            copy->copyFromTexture(
                textures[Output], 0, 0, MTL::Origin::Make(0, 0, 0), MTL::Size::Make(layout.width, layout.height, 1),
                bridge, slotBase + layout.offsets[Output], layout.rows[Output], layout.rows[Output] * layout.height);
            copy->endEncoding();
            command->endCommandBuffer();
            struct Completion {
                std::mutex mutex;
                std::condition_variable cv;
                bool done = false, error = false;
            } completion;
            auto *commit = MTL4::CommitOptions::alloc()->init();
            commit->addFeedbackHandler([&completion](MTL4::CommitFeedback *f) {
                std::lock_guard lock(completion.mutex);
                completion.error = f->error() != nullptr;
                completion.done = true;
                completion.cv.notify_one();
            });
            const MTL4::CommandBuffer *batch[] = {command};
            context.queue()->commit(batch, 1, commit);
            commit->release();
            context.queue()->signalEvent(done, request.ticket);
            bool complete = done->waitUntilSignaledValue(request.ticket, FrameTimeoutMs);
            {
                std::unique_lock lock(completion.mutex);
                complete = completion.cv.wait_for(lock, std::chrono::milliseconds(FrameTimeoutMs),
                                                  [&] { return completion.done; }) &&
                           complete && !completion.error;
            }
            if (!complete)
                std::_Exit(87); // No stack callback or allocator can outlive a timed-out GPU submission.
            reply.ticket = request.ticket;
            reply.deviceBytes = context.device()->currentAllocatedSize();
            reply.physicalFootprint = processFootprint();
            reply.gpuAllocations = context.memory().allocationCount();
            previous = request.ticket;
            const bool sent = transfer(3, &reply, sizeof(reply), true, FrameTimeoutMs);
            pool->release();
            if (!sent)
                break;
        }
        command->release();
        allocator->release();
        done->release();
        fence->release();
        scaler.reset();
        for (auto *t : textures)
            context.memory().release(t, MemoryCategory::RenderTargets);
        context.memory().release(bridge, MemoryCategory::Other);
        context.memory().releaseCompleted(~u64{0});
    } catch (const std::exception &e) {
        LOG_ERROR("MetalFX worker failed: %s", e.what());
        exitCode = 2;
    }
    close(3);
    return exitCode;
}
} // namespace phosphor
