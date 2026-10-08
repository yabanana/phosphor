// CPU-only probe of the original table generator. Not linked to the renderer.
#include "renderer/stochastic_sampling.h"
#include <array>
#include <bit>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: stbn_spectral_generate FRESH_OUTPUT_DIRECTORY");
        const std::filesystem::path directory(argv[1]);
        if (std::filesystem::exists(directory)) throw std::runtime_error("output directory must not already exist");
        std::filesystem::create_directories(directory);
        std::ofstream metadata(directory / "generation.json");
        if (!metadata) throw std::runtime_error("cannot write generation metadata");
        metadata << "{\n  \"generator_version\": " << phosphor::di::STOCHASTIC_GENERATOR_VERSION
                 << ",\n  \"compiler\": " << std::quoted(__VERSION__)
                 << ",\n  \"config\": {\"width\":8,\"height\":8,\"frames\":16,\"dimensions\":64,"
                    "\"sigma_spatial\":1.9,\"sigma_temporal\":1.9,\"max_relaxation_swaps\":4096},\n  \"runs\": [\n";
        for (unsigned seed=0;seed<8;++seed) {
            phosphor::di::StbnConfig config;
            config.width=8; config.height=8; config.frames=16; config.dimensions=64;
            config.sigmaSpatial=1.9f; config.sigmaTemporal=1.9f; config.maxRelaxationSwaps=4096; config.seed=seed;
            const auto start=std::chrono::steady_clock::now();
            const auto mask=phosphor::di::generateStbn(config);
            const auto end=std::chrono::steady_clock::now();
            const double milliseconds=std::chrono::duration<double,std::milli>(end-start).count();
            const auto filename="seed-"+std::to_string(seed)+".u32le";
            std::ofstream binary(directory / filename,std::ios::binary);
            if (!binary) throw std::runtime_error("cannot write ranks");
            if constexpr(std::endian::native==std::endian::little) {
                binary.write(reinterpret_cast<const char*>(mask.ranks.data()),mask.ranks.size()*sizeof(phosphor::u32));
            } else {
                for(auto rank:mask.ranks) {
                    const std::array<char,4> bytes{char(rank),char(rank>>8),char(rank>>16),char(rank>>24)};
                    binary.write(bytes.data(),bytes.size());
                }
            }
            binary.close();
            if (!binary) throw std::runtime_error("rank write failed");
            metadata << (seed?",\n":"") << "    {\"seed\":" << seed << ",\"file\":" << std::quoted(filename)
                     << ",\"generation_ms\":" << std::setprecision(12) << milliseconds
                     << ",\"relaxation_converged\":" << (mask.relaxationConverged?"true":"false") << "}";
            metadata.flush();
            std::cout << "seed " << seed << " generation_ms " << milliseconds << " converged " << mask.relaxationConverged << '\n' << std::flush;
        }
        metadata << "\n  ]\n}\n";
        metadata.close();
        if (!metadata) throw std::runtime_error("metadata write failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
