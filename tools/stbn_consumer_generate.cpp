// Calls the actual frozen C++ sample() on previously generated rank tables.
#include "renderer/stochastic_sampling.h"
#include <bit>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc,char** argv) {
    static_assert(std::endian::native==std::endian::little && sizeof(float)==4);
    try {
        if(argc!=3)throw std::runtime_error("usage: stbn_consumer_generate RAW_RANK_DIRECTORY FRESH_OUTPUT_DIRECTORY");
        const std::filesystem::path input(argv[1]),output(argv[2]);
        if(std::filesystem::exists(output))throw std::runtime_error("output directory must be fresh");
        std::filesystem::create_directories(output);
        std::ofstream metadata(output/"consumer.json");
        if(!metadata)throw std::runtime_error("cannot create consumer metadata");
        metadata << "{\"generator_version\":" << phosphor::di::STOCHASTIC_GENERATOR_VERSION
                 << ",\"compiler\":" << std::quoted(__VERSION__) << ",\"runs\":[\n";
        bool first=true;
        for(unsigned seed=0;seed<8;++seed) {
            phosphor::di::StbnMask mask;
            mask.config.seed=seed;
            if(mask.config.width!=8 || mask.config.height!=8 || mask.config.frames!=16 || mask.config.dimensions!=64)
                throw std::runtime_error("frozen sampler configuration differs from protocol");
            mask.ranks.resize(64*16*8*8);
            const auto file=input/("seed-"+std::to_string(seed)+".u32le");
            if(std::filesystem::file_size(file)!=mask.ranks.size()*4)throw std::runtime_error("rank byte count differs");
            std::ifstream source(file,std::ios::binary);
            source.read(reinterpret_cast<char*>(mask.ranks.data()),mask.ranks.size()*4);
            if(!source)throw std::runtime_error("cannot read original ranks");
            for(unsigned block=0;block<4;++block) {
                std::vector<float> values(mask.ranks.size());
                for(unsigned dimension=0;dimension<64;++dimension)for(unsigned t=0;t<16;++t)
                    for(unsigned y=0;y<8;++y)for(unsigned x=0;x<8;++x)
                        values[((dimension*16+t)*8+y)*8+x]=mask.sample(x,y,block*16+t,dimension);
                const auto name="seed-"+std::to_string(seed)+"-block-"+std::to_string(block)+".f32le";
                std::ofstream data(output/name,std::ios::binary);
                data.write(reinterpret_cast<const char*>(values.data()),values.size()*4);data.close();
                if(!data)throw std::runtime_error("sample write failed");
                metadata << (first?"":",\n") << "{\"seed\":" << seed << ",\"block\":" << block
                         << ",\"file\":" << std::quoted(name) << "}";
                first=false;
            }
        }
        metadata << "\n]}\n";metadata.close();
        if(!metadata)throw std::runtime_error("consumer metadata write failed");
        std::cout << "32 CPU sample blocks written; original rank tables reused without regeneration\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
