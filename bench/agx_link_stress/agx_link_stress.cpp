// F3.2/F15 AGX linked-intersection compile stress: a MEASUREMENT TOOL, not engine code.
//
// Question: does a valid concurrent use of one thread-safe MTL4::Compiler crash
// inside the driver when several compute pipelines that statically link the
// same intersection function (rt_alpha_generic) are compiled on many threads,
// under MTL_SHADER_VALIDATION (the crash report of corpus case
// negative-f14-lut: AGX UserIntersectionProgram::VariantSetEntry::signalPending
// from createAirLinkedIntersectionVariantSets on a phosphor-compile thread)?
//
// The tool compiles the way PipelineCache does (same descriptor shape: a
// specialised-with-no-constants main function, a library descriptor for the
// linked function, one shared MTL4::Compiler, an autorelease pool per job) but
// owns nothing else.  Each iteration loads a fresh MTL::Library from the
// application metallib and compiles every requested kernel once, spread over
// --threads worker threads.  tools/agx_link_stress.py runs batches of it in
// child processes so that a crash is counted instead of fatal.
//
// Usage: agx_link_stress --metallib PATH [--threads T] [--iters N]
//        [--kernels a,b,c] [--fresh-library 0|1] [--compiler-per-thread 0|1]
//        [--serialize 0|1] [--link 0|1] [--verbose]
// --salted K compiles K synthetic kernels (self-contained MSL, unique source per iteration = cold compile).
// --serialize 1 holds one mutex around every compile (the proposed mitigation).
// --link 0 compiles the same kernels without the static linking descriptor
// (they cannot all compile that way: kernels reading the IFT still do, the
// function table is just never populated; used as a control).
// Exit: 0 all compiled, 1 some compile failed, 2 usage, signal on a crash.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

NS::String* str(const char* s) { return NS::String::string(s, NS::UTF8StringEncoding); }

struct Options {
    std::string metallib;
    unsigned threads         = 8;
    unsigned iters           = 4;
    bool freshLibrary        = true;
    bool compilerPerThread   = false;
    bool serialize           = false;
    bool link                = true;
    bool verbose             = false;
    bool bind                = false; // the consumer also binds each finished pipeline on a compute encoder (render thread's setComputePipelineState)
    bool consume             = false; // a consumer thread (the render thread's role) uses finished pipelines meanwhile
    unsigned plain           = 0;     // plus this many salted compute kernels with no linked function
    unsigned salted          = 0; // >0: compile this many salted synthetic kernels from MSL source (cold every iteration)
    std::vector<std::string> kernels = {"ao_rtao", "fog_inject_rt", "reflection_capture_rt", "reflection_rt",
                                        "reflection_rt_lobe", "rt_trace_rays", "shadow_sun_rt", "ddgi_trace",
                                        "gi_candidates", "gi_temporal", "gi_spatial", "gi_shade",
                                        "light_cluster_shade_rt", "light_cluster_shade_rt_lobe", "restir_di_shade"};
};

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

/// Self-contained MSL: one intersection function and `kernels` kernels that
/// trace through an IFT.  `salt` changes the source, so the driver and the
/// shader cache see new content (a cold compile) on every call.
std::string saltedSource(unsigned kernels, unsigned plain, unsigned salt) {
    std::string src =
        "#include <metal_stdlib>\nusing namespace metal; using namespace raytracing;\n"
        "struct P { uint a; uint b; float t; };\n"
        "[[intersection(triangle, triangle_data, instancing)]]\n"
        "bool alpha_generic(uint primitive [[primitive_id]], uint slot [[user_instance_id]], float2 bary [[barycentric_coord]],\n"
        "                   float distance [[distance]], ray_data P& payload [[payload]], const device float* tab [[buffer(0)]]) {\n"
        "    ++payload.a; payload.t = distance;\n"
        "    return tab[(slot + primitive + " + std::to_string(salt) + "u) & 3u] > bary.x;\n}\n";
    for (unsigned k = 0; k < kernels; ++k) {
        src += "kernel void salted_" + std::to_string(k) +
               "(instance_acceleration_structure as [[buffer(0)]], const device float4* rays [[buffer(1)]], device float4* out [[buffer(2)]],\n"
               "  intersection_function_table<triangle_data, instancing> ift [[buffer(4)]], uint tid [[thread_position_in_grid]]) {\n"
               "    P payload{}; intersector<triangle_data, instancing> i; i.assume_geometry_type(geometry_type::triangle);\n"
               "    const float4 r = rays[tid] * " + std::to_string(k + 1) + ".0f + " + std::to_string(salt % 977) + ".0f;\n"
               "    const auto res = i.intersect(ray(r.xyz, normalize(float3(1.0f, r.w, 0.25f)), 0.0f, 100.0f), as, 0xFFu, ift, payload);\n"
               "    out[tid] = float4(res.distance, float(payload.a), payload.t, float(res.user_instance_id));\n}\n";
    }
    for (unsigned k = 0; k < plain; ++k) {
        src += "kernel void plain_" + std::to_string(k) +
               "(const device float4* in [[buffer(0)]], device float4* out [[buffer(1)]], uint tid [[thread_position_in_grid]]) {\n"
               "    float4 v = in[tid]; for (uint n = 0; n < 24u; ++n) v = sin(v * " + std::to_string(k + 2) + ".0f + " +
               std::to_string(salt % 613) + ".0f) + v.wzyx;\n    out[tid] = v;\n}\n";
    }
    return src;
}

/// Same descriptor shape as PipelineCache::buildDescriptor (compute branch).
MTL4::ComputePipelineDescriptor* buildDescriptor(const std::string& kernel, MTL::Library* library, bool link, const char* linked) {
    auto* d = MTL4::ComputePipelineDescriptor::alloc()->init();
    d->setLabel(str(kernel.c_str()));
    auto* main = MTL4::LibraryFunctionDescriptor::alloc()->init();
    main->setLibrary(library);
    main->setName(str(kernel.c_str()));
    auto* values = MTL::FunctionConstantValues::alloc()->init();
    auto* spec = MTL4::SpecializedFunctionDescriptor::alloc()->init();
    spec->setFunctionDescriptor(main);
    spec->setConstantValues(values);
    values->release();
    main->release();
    d->setComputeFunctionDescriptor(spec);
    spec->release();
    if (link) {
        auto* fn = MTL4::LibraryFunctionDescriptor::alloc()->init();
        fn->setLibrary(library);
        fn->setName(str(linked));
        NS::Object* functions[] = {fn};
        auto* sl = MTL4::StaticLinkingDescriptor::alloc()->init();
        sl->setFunctionDescriptors(NS::Array::array(functions, 1));
        d->setStaticLinkingDescriptor(sl);
        sl->release();
        fn->release();
    }
    return d;
}

MTL4::Compiler* makeCompiler(MTL::Device* device) {
    auto* cd = MTL4::CompilerDescriptor::alloc()->init();
    cd->setLabel(str("agx_link_stress compiler"));
    NS::Error* error = nullptr;
    MTL4::Compiler* compiler = device->newCompiler(cd, &error);
    cd->release();
    if (!compiler) {
        std::fprintf(stderr, "newCompiler failed: %s\n", error ? error->localizedDescription()->utf8String() : "?");
        std::exit(2);
    }
    return compiler;
}

} // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--metallib") o.metallib = next();
        else if (a == "--threads") o.threads = static_cast<unsigned>(std::atoi(next()));
        else if (a == "--iters") o.iters = static_cast<unsigned>(std::atoi(next()));
        else if (a == "--kernels") o.kernels = split(next());
        else if (a == "--fresh-library") o.freshLibrary = std::atoi(next()) != 0;
        else if (a == "--compiler-per-thread") o.compilerPerThread = std::atoi(next()) != 0;
        else if (a == "--serialize") o.serialize = std::atoi(next()) != 0;
        else if (a == "--link") o.link = std::atoi(next()) != 0;
        else if (a == "--verbose") o.verbose = true;
        else if (a == "--bind") o.bind = std::atoi(next()) != 0;
        else if (a == "--consume") o.consume = std::atoi(next()) != 0;
        else if (a == "--plain") o.plain = static_cast<unsigned>(std::atoi(next()));
        else if (a == "--salted") o.salted = static_cast<unsigned>(std::atoi(next()));
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (o.metallib.empty() || o.threads == 0) { std::fprintf(stderr, "usage: see the file header\n"); return 2; }

    auto* mainPool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) { std::fprintf(stderr, "no Metal device\n"); return 2; }
    NS::Error* error = nullptr;
    NS::URL* url = NS::URL::fileURLWithPath(str(o.metallib.c_str()));
    MTL::Library* shared = o.freshLibrary ? nullptr : device->newLibrary(url, &error);
    if (!o.freshLibrary && !shared) { std::fprintf(stderr, "newLibrary failed\n"); return 2; }
    MTL4::Compiler* sharedCompiler = o.compilerPerThread ? nullptr : makeCompiler(device);
    std::mutex serial;

    std::atomic<unsigned> failures{0}, compiled{0};
    for (unsigned iter = 0; iter < o.iters; ++iter) {
        MTL::Library* library = shared;
        std::vector<std::string> names = o.kernels;
        const char* linked = "rt_alpha_generic";
        if (o.salted) {
            names.clear();
            linked = "alpha_generic";
            for (unsigned k = 0; k < o.salted; ++k) names.push_back("salted_" + std::to_string(k));
            for (unsigned k = 0; k < o.plain; ++k) names.push_back("plain_" + std::to_string(k));
            auto* opts = MTL::CompileOptions::alloc()->init();
            library = device->newLibrary(str(saltedSource(o.salted, o.plain, iter * 7919u + static_cast<unsigned>(getpid()) * 31u).c_str()), opts, &error);
            opts->release();
            if (!library) {
                std::fprintf(stderr, "salted newLibrary failed: %s\n", error ? error->localizedDescription()->utf8String() : "?");
                return 2;
            }
        } else if (o.freshLibrary) {
            library = device->newLibrary(url, &error);
            if (!library) { std::fprintf(stderr, "newLibrary failed\n"); return 2; }
        }
        std::atomic<size_t> cursor{0};
        std::mutex handoffMutex;
        std::vector<MTL::ComputePipelineState*> handoff;
        std::atomic<bool> compilesDone{false};
        std::atomic<unsigned> consumed{0};
        // The render thread's role (RtConsumer::prepare, AccelerationStructures::updateIft): resolve the
        // linked function handle of a finished pipeline and build its intersection function table, while
        // other workers are still compiling pipelines linked to the same function.
        std::thread consumer;
        if (o.consume) consumer = std::thread([&] {
            MTL4::CommandAllocator* allocator = o.bind ? device->newCommandAllocator() : nullptr;
            MTL4::CommandBuffer* cb = o.bind ? device->newCommandBuffer() : nullptr;
            for (;;) {
                MTL::ComputePipelineState* pso = nullptr;
                {
                    std::lock_guard<std::mutex> lock(handoffMutex);
                    if (!handoff.empty()) { pso = handoff.back(); handoff.pop_back(); }
                }
                if (!pso) {
                    if (compilesDone.load()) break;
                    std::this_thread::yield();
                    continue;
                }
                auto* pool = NS::AutoreleasePool::alloc()->init();
                auto* handle = pso->functionHandle(str(linked));
                auto* desc = MTL::IntersectionFunctionTableDescriptor::alloc()->init();
                desc->setFunctionCount(1);
                auto* table = pso->newIntersectionFunctionTable(desc);
                desc->release();
                if (table && handle) table->setFunction(handle, 0);
                else std::fprintf(stderr, "consumer: no handle/table\n");
                if (table && o.bind) {
                    cb->beginCommandBuffer(allocator);
                    auto* enc = cb->computeCommandEncoder();
                    enc->setComputePipelineState(pso);
                    enc->endEncoding();
                    cb->endCommandBuffer();
                    allocator->reset();
                }
                if (table) table->release();
                pso->release();
                ++consumed;
                pool->release();
            }
            if (cb) cb->release();
            if (allocator) allocator->release();
        });
        const auto start = Clock::now();
        auto worker = [&](unsigned id) {
            MTL4::Compiler* compiler = o.compilerPerThread ? makeCompiler(device) : sharedCompiler;
            for (;;) {
                const size_t k = cursor.fetch_add(1);
                if (k >= names.size()) break;
                auto* pool = NS::AutoreleasePool::alloc()->init();
                const bool linkThis = o.link && names[k].rfind("plain_", 0) != 0;
                auto* d = buildDescriptor(names[k], library, linkThis, linked);
                NS::Error* err = nullptr;
                MTL::ComputePipelineState* pso = nullptr;
                {
                    std::unique_lock<std::mutex> lock(serial, std::defer_lock);
                    if (o.serialize) lock.lock();
                    pso = compiler->newComputePipelineState(d, nullptr, &err);
                }
                if (pso) {
                    ++compiled;
                    if (o.consume && linkThis) {
                        std::lock_guard<std::mutex> lock(handoffMutex);
                        handoff.push_back(pso);
                    } else {
                        pso->release();
                    }
                } else {
                    ++failures;
                    std::fprintf(stderr, "iter %u thread %u: %s failed: %s\n", iter, id, names[k].c_str(),
                                 err ? err->localizedDescription()->utf8String() : "?");
                }
                d->release();
                pool->release();
            }
            if (o.compilerPerThread) compiler->release();
        };
        std::vector<std::thread> threads;
        for (unsigned t = 0; t < o.threads; ++t) threads.emplace_back(worker, t);
        for (auto& t : threads) t.join();
        compilesDone = true;
        if (consumer.joinable()) consumer.join();
        if (o.verbose)
            std::printf("iter %u: %zu kernels, %u threads, %.1f ms\n", iter, names.size(), o.threads,
                        std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        if (o.freshLibrary || o.salted) library->release();
    }
    std::printf("DONE compiled %u failed %u\n", compiled.load(), failures.load());
    if (shared) shared->release();
    if (sharedCompiler) sharedCompiler->release();
    mainPool->release();
    return failures.load() ? 1 : 0;
}
