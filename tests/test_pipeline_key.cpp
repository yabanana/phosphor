#include "pipeline/pipeline_desc.h"
#include "pipeline/pipeline_key.h"

#include <doctest/doctest.h>

#include <cstring>
#include <string>
#include <utility>

using namespace phosphor;
using namespace phosphor::pipe;

namespace {

PipelineDesc baseDesc() {
    PipelineDesc d;
    d.label        = "forward";
    d.functions[0] = "forward_vs";
    d.functions[1] = "forward_fs";
    d.constant(0, ConstantType::UInt, 7).constant(1, ConstantType::Bool, 1);
    d.output(0, rg::Format::BGRA8Srgb);
    return d;
}

} // namespace

TEST_CASE("pipeline key: canonical string format") {
    CHECK(canonicalString(baseDesc()) == "R|forward_vs|forward_fs|c0:u=7,c1:b=1|o0=BGRA8Srgb/none/F");

    PipelineDesc c;
    c.kind         = PipelineKind::Compute;
    c.functions[0] = "cull_cs";
    c.constant(3, ConstantType::Int, static_cast<u32>(-5));
    float f = 1.0f;
    u32 bits;
    std::memcpy(&bits, &f, sizeof(bits));
    c.constant(4, ConstantType::Float, bits);
    CHECK(canonicalString(c) == "C|cull_cs||c3:i=-5,c4:f=0x3f800000|");

    PipelineDesc b = baseDesc();
    b.output(1, rg::Format::RGBA16Float, ColorOutput::Blend::AlphaOver);
    b.color[1].writeMask = 0x7;
    CHECK(canonicalString(b) == "R|forward_vs|forward_fs|c0:u=7,c1:b=1|o0=BGRA8Srgb/none/F,o1=RGBA16Float/over/7");
}

TEST_CASE("pipeline key: golden hashes (changing them invalidates harvested data)") {
    // Computed once from the algorithm documented in pipeline_key.cpp.  If one of
    // these fails, the canonical format or the hash changed: every stored key
    // (harvest lists, archives keyed by them) is invalid.
    CHECK(hashBytes("", 0) == 0xf52a15e9a9b5e89bull);
    CHECK(hashBytes("phosphor", 8) == 0x8d18332298a53e4aull);
    CHECK(pipelineKey(baseDesc()) == 0xc7bd0d90cf603f89ull);
    CHECK(pipelineKey(baseDesc(), 42) == 0xae20c6cf2faab8e2ull);
    CHECK(pipelineKey(baseDesc().generic()) == 0x47f7d53e3a35f0b9ull);
}

TEST_CASE("pipeline key: deterministic") {
    CHECK(pipelineKey(baseDesc()) == pipelineKey(baseDesc()));
    CHECK(hashBytes("abc", 3) == hashBytes("abc", 3));
    CHECK(hashBytes("abc", 3) != hashBytes("abd", 3));
    CHECK(hashBytes("abc", 3, 1) != hashBytes("abc", 3, 0));
}

TEST_CASE("pipeline key: sensitivity") {
    const PipelineKey ref = pipelineKey(baseDesc());

    SUBCASE("label is ignored") {
        PipelineDesc d = baseDesc();
        d.label        = "something else";
        CHECK(pipelineKey(d) == ref);
    }
    SUBCASE("kind") {
        PipelineDesc d = baseDesc();
        d.kind         = PipelineKind::Compute;
        CHECK(pipelineKey(d) != ref);
    }
    SUBCASE("functions") {
        PipelineDesc d = baseDesc();
        d.functions[0] = "other_vs";
        CHECK(pipelineKey(d) != ref);
        d = baseDesc();
        d.functions[1] = "other_fs";
        CHECK(pipelineKey(d) != ref);
    }
    SUBCASE("constant index, type, value, order") {
        PipelineDesc d = baseDesc();
        d.constants[0].index = 9;
        CHECK(pipelineKey(d) != ref);
        d = baseDesc();
        d.constants[0].type = ConstantType::Int;
        CHECK(pipelineKey(d) != ref);
        d = baseDesc();
        d.constants[0].bits = 8;
        CHECK(pipelineKey(d) != ref);
        d = baseDesc();
        std::swap(d.constants[0], d.constants[1]);
        CHECK(pipelineKey(d) != ref);
    }
    SUBCASE("colour output format, blend, write mask") {
        PipelineDesc d = baseDesc();
        d.output(0, rg::Format::RGBA16Float);
        CHECK(pipelineKey(d) != ref);
        d = baseDesc();
        d.output(0, rg::Format::BGRA8Srgb, ColorOutput::Blend::AlphaOver);
        CHECK(pipelineKey(d) != ref);
        d = baseDesc();
        d.color[0].writeMask = 0x3;
        CHECK(pipelineKey(d) != ref);
    }
    SUBCASE("extra attachment") {
        PipelineDesc d = baseDesc();
        d.output(1, rg::Format::R32Float);
        CHECK(pipelineKey(d) != ref);
    }
}

TEST_CASE("pipeline desc: builders") {
    PipelineDesc d;
    CHECK(d.isGeneric());
    CHECK_FALSE(d.isFlexible());
    d.output(2, rg::Format::R8Unorm);
    CHECK(d.colorCount == 3);
    d.constant(1, ConstantType::UInt, 4);
    CHECK(d.constantCount == 1);
    CHECK_FALSE(d.isGeneric());
}

TEST_CASE("pipeline desc: flexible") {
    const PipelineDesc a = baseDesc();
    CHECK_FALSE(a.isFlexible());
    const PipelineDesc fa = a.flexible();
    CHECK(fa.isFlexible());
    CHECK(fa.constantCount == a.constantCount);
    CHECK(canonicalString(fa) == "R|forward_vs|forward_fs|c0:u=7,c1:b=1|o0=unspecialized");

    // The format kept in the struct is irrelevant once unspecialized.
    PipelineDesc b = baseDesc();
    b.output(0, rg::Format::RGBA16Float, ColorOutput::Blend::AlphaOver);
    CHECK(pipelineKey(b.flexible()) == pipelineKey(fa));
    CHECK(pipelineKey(b) != pipelineKey(a));

    // Unused attachment slots stay unused.
    PipelineDesc gap = baseDesc();
    gap.output(2, rg::Format::R32Float);
    const PipelineDesc fg = gap.flexible();
    CHECK(canonicalString(fg) == "R|forward_vs|forward_fs|c0:u=7,c1:b=1|o0=unspecialized,o2=unspecialized");
    CHECK(fg.color[1].unspecialized == false);
}

TEST_CASE("pipeline desc: generic and salt") {
    const PipelineDesc a = baseDesc();
    const PipelineDesc g = a.generic();
    CHECK(g.isGeneric());
    CHECK(g.constantCount == 0);
    CHECK(g.functions == a.functions);
    CHECK(g.color[0] == a.color[0]);
    CHECK(canonicalString(g) == "R|forward_vs|forward_fs||o0=BGRA8Srgb/none/F");

    // Salt only affects specialised variants.
    CHECK(pipelineKey(g, 0) == pipelineKey(g, 1234));
    CHECK(pipelineKey(a, 0) != pipelineKey(a, 1234));
    CHECK(pipelineKey(a, 1) != pipelineKey(a, 2));
    CHECK(pipelineKey(a, 0) == pipelineKey(a));
    CHECK(pipelineKey(g) != pipelineKey(a));
}

namespace {

PipelineDesc meshDesc() {
    PipelineDesc d;
    d.kind         = PipelineKind::Mesh;
    d.label        = "meshlet";
    d.functions[0] = "meshlet_object";
    d.functions[1] = "meshlet_mesh";
    d.functions[2] = "forward_fs";
    d.constant(0, ConstantType::UInt, 7);
    d.output(0, rg::Format::BGRA8Srgb);
    d.mesh = {32, 128, 384, 32};
    return d;
}

} // namespace

TEST_CASE("pipeline key: mesh canonical string") {
    CHECK(canonicalString(meshDesc()) ==
          "M|meshlet_object|meshlet_mesh|forward_fs|c0:u=7|o0=BGRA8Srgb/none/F|mesh=32,128,384,32");

    // No object stage: the empty function keeps its slot.
    PipelineDesc d = meshDesc();
    d.functions[0] = "";
    d.mesh.objectThreads = 0;
    CHECK(canonicalString(d) == "M||meshlet_mesh|forward_fs|c0:u=7|o0=BGRA8Srgb/none/F|mesh=0,128,384,32");

    // Generic variant keeps the mesh limits.
    CHECK(canonicalString(meshDesc().generic()) ==
          "M|meshlet_object|meshlet_mesh|forward_fs||o0=BGRA8Srgb/none/F|mesh=32,128,384,32");
}

TEST_CASE("pipeline key: mesh sensitivity") {
    const PipelineKey ref = pipelineKey(meshDesc());
    CHECK(pipelineKey(meshDesc()) == ref);

    SUBCASE("label is ignored") {
        PipelineDesc d = meshDesc();
        d.label        = "other";
        CHECK(pipelineKey(d) == ref);
    }
    SUBCASE("each function") {
        for (u32 i = 0; i < 3; ++i) {
            PipelineDesc d = meshDesc();
            d.functions[i] = "changed";
            CHECK_MESSAGE(pipelineKey(d) != ref, "function " << i);
        }
    }
    SUBCASE("each limit") {
        PipelineDesc d = meshDesc();
        d.mesh.objectThreads++;
        CHECK(pipelineKey(d) != ref);
        d = meshDesc();
        d.mesh.meshThreads++;
        CHECK(pipelineKey(d) != ref);
        d = meshDesc();
        d.mesh.payloadBytes++;
        CHECK(pipelineKey(d) != ref);
        d = meshDesc();
        d.mesh.meshGroups++;
        CHECK(pipelineKey(d) != ref);
    }
    SUBCASE("function and limit values do not alias") {
        // The same numbers in another slot are another pipeline.
        PipelineDesc d = meshDesc();
        d.mesh = {128, 32, 384, 32};
        CHECK(pipelineKey(d) != ref);
    }
    SUBCASE("constants and outputs still count") {
        PipelineDesc d = meshDesc();
        d.constants[0].bits = 8;
        CHECK(pipelineKey(d) != ref);
        d = meshDesc();
        d.output(0, rg::Format::RGBA16Float);
        CHECK(pipelineKey(d) != ref);
    }
    SUBCASE("salt") {
        CHECK(pipelineKey(meshDesc(), 5) != ref);
        CHECK(pipelineKey(meshDesc().generic(), 5) == pipelineKey(meshDesc().generic(), 0));
    }
}

TEST_CASE("pipeline key: mesh differs from render with the same first two functions") {
    PipelineDesc r;
    r.functions[0] = "meshlet_object";
    r.functions[1] = "meshlet_mesh";
    r.constant(0, ConstantType::UInt, 7);
    r.output(0, rg::Format::BGRA8Srgb);

    PipelineDesc m = meshDesc();
    m.functions[2] = "";
    m.mesh         = {};
    CHECK(canonicalString(r) != canonicalString(m));
    CHECK(pipelineKey(r) != pipelineKey(m));
    CHECK(canonicalString(r) == "R|meshlet_object|meshlet_mesh|c0:u=7|o0=BGRA8Srgb/none/F");
}

TEST_CASE("pipeline key: render and compute strings and keys are unchanged by the mesh kind") {
    CHECK(canonicalString(baseDesc()).find("mesh=") == std::string::npos);
    CHECK(canonicalString(baseDesc()) == "R|forward_vs|forward_fs|c0:u=7,c1:b=1|o0=BGRA8Srgb/none/F");
    CHECK(pipelineKey(baseDesc()) == 0xc7bd0d90cf603f89ull);
    PipelineDesc c;
    c.kind         = PipelineKind::Compute;
    c.functions[0] = "cull_cs";
    CHECK(canonicalString(c) == "C|cull_cs|||");
    // The third function and the mesh limits are ignored outside the mesh kind.
    PipelineDesc r = baseDesc();
    r.functions[2] = "stray";
    r.mesh         = {1, 2, 3, 4};
    CHECK(pipelineKey(r) == pipelineKey(baseDesc()));
}
