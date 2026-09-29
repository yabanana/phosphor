// variant_gen -- reads shaders/variants.def and writes the generated variant
// tables (C++ and MSL).  Plain C++20, no dependencies; host tool of the build.
//
//   variant_gen <variants.def> <output-dir>
//
// Writes <output-dir>/pipeline/forward_variants.generated.h and
// <output-dir>/pipeline/forward_variants.generated.metal.h, only when their
// content changed (no needless rebuilds).  Exit code 1 on a malformed file.
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Axis {
    std::string name;
    bool        isBool = false;
    unsigned    index = 0;
    unsigned    minValue = 0;
    unsigned    maxValue = 0;
    unsigned    generic = 0;
    bool        runtime = false;
    unsigned    range() const { return maxValue - minValue + 1; }
};

struct Reserved {
    std::string name;
    unsigned    index = 0;
};

[[noreturn]] void fail(const std::string& file, int line, const std::string& msg) {
    std::fprintf(stderr, "variant_gen: %s:%d: error: %s\n", file.c_str(), line, msg.c_str());
    std::exit(1);
}

std::string camel(const std::string& name) {
    std::string out;
    bool upper = true;
    for (char c : name) {
        if (c == '_') { upper = true; continue; }
        out += upper ? char(std::toupper(c)) : char(std::tolower(c));
        upper = false;
    }
    return out;
}

bool parseUnsigned(const std::string& s, unsigned& out) {
    if (s.empty() || s.size() > 9) return false;
    for (char c : s) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    out = static_cast<unsigned>(std::stoul(s));
    return true;
}

bool parseValue(const std::string& s, bool isBool, unsigned& out) {
    if (isBool && s == "true")  { out = 1; return true; }
    if (isBool && s == "false") { out = 0; return true; }
    return parseUnsigned(s, out);
}

bool validName(const std::string& n) {
    if (n.empty() || !std::isupper(static_cast<unsigned char>(n[0]))) return false;
    for (char c : n) {
        if (!std::isupper(static_cast<unsigned char>(c)) && !std::isdigit(static_cast<unsigned char>(c)) && c != '_') return false;
    }
    return true;
}

bool writeIfChanged(const std::filesystem::path& path, const std::string& content) {
    {
        std::ifstream in(path, std::ios::binary);
        if (in) {
            std::stringstream ss;
            ss << in.rdbuf();
            if (ss.str() == content) return true;
        }
    }
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << content;
    out.flush();
    if (!out) {
        std::fprintf(stderr, "variant_gen: cannot write %s\n", path.string().c_str());
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: variant_gen <variants.def> <output-dir>\n");
        return 2;
    }
    const std::string defPath = argv[1];
    std::ifstream in(defPath);
    if (!in) {
        std::fprintf(stderr, "variant_gen: cannot open %s\n", defPath.c_str());
        return 1;
    }

    std::vector<Axis> axes;
    std::vector<Reserved> reserved;
    std::set<std::string> names;
    std::set<unsigned> indices;

    std::string raw;
    int lineNo = 0;
    while (std::getline(in, raw)) {
        ++lineNo;
        if (const auto hash = raw.find('#'); hash != std::string::npos) raw.erase(hash);
        std::istringstream ls(raw);
        std::vector<std::string> tok;
        for (std::string t; ls >> t;) tok.push_back(t);
        if (tok.empty()) continue;

        if (tok[0] == "axis") {
            if (tok.size() != 7) fail(defPath, lineNo, "expected: axis <NAME> <bool|uint> <index> <min> <max> <generic|runtime>");
            Axis a;
            a.name = tok[1];
            if (!validName(a.name)) fail(defPath, lineNo, "invalid axis name '" + a.name + "' (want [A-Z][A-Z0-9_]*)");
            if (!names.insert(a.name).second) fail(defPath, lineNo, "duplicate name '" + a.name + "'");
            if (tok[2] == "bool") a.isBool = true;
            else if (tok[2] != "uint") fail(defPath, lineNo, "type must be bool or uint, got '" + tok[2] + "'");
            if (!parseUnsigned(tok[3], a.index) || a.index > 30) fail(defPath, lineNo, "constant index must be 0..30 (31 is the salt)");
            if (!indices.insert(a.index).second) fail(defPath, lineNo, "duplicate constant index " + tok[3]);
            if (!parseValue(tok[4], a.isBool, a.minValue)) fail(defPath, lineNo, "bad min '" + tok[4] + "'");
            if (!parseValue(tok[5], a.isBool, a.maxValue)) fail(defPath, lineNo, "bad max '" + tok[5] + "'");
            if (a.minValue > a.maxValue) fail(defPath, lineNo, "min > max");
            if (a.isBool && (a.minValue != 0 || a.maxValue != 1)) fail(defPath, lineNo, "bool axes must range over 0..1 (false true)");
            if (tok[6] == "runtime") {
                a.runtime = true;
            } else {
                if (!parseValue(tok[6], a.isBool, a.generic)) fail(defPath, lineNo, "bad generic value '" + tok[6] + "'");
                if (a.generic < a.minValue || a.generic > a.maxValue) fail(defPath, lineNo, "generic value outside [min, max]");
            }
            axes.push_back(a);
        } else if (tok[0] == "reserved") {
            if (tok.size() != 4) fail(defPath, lineNo, "expected: reserved <NAME> uint <index>");
            Reserved r;
            r.name = tok[1];
            if (!validName(r.name)) fail(defPath, lineNo, "invalid name '" + r.name + "'");
            if (!names.insert(r.name).second) fail(defPath, lineNo, "duplicate name '" + r.name + "'");
            if (tok[2] != "uint") fail(defPath, lineNo, "reserved constants must be uint");
            if (!parseUnsigned(tok[3], r.index) || r.index > 31) fail(defPath, lineNo, "constant index must be 0..31");
            if (!indices.insert(r.index).second) fail(defPath, lineNo, "duplicate constant index " + tok[3]);
            reserved.push_back(r);
        } else {
            fail(defPath, lineNo, "unknown directive '" + tok[0] + "' (want axis or reserved)");
        }
    }

    if (axes.empty()) fail(defPath, lineNo, "no axis defined");
    if (axes.size() > 7) fail(defPath, lineNo, "too many axes (PipelineDesc holds 8 constants, one is the salt)");
    const Reserved* salt = nullptr;
    for (const auto& r : reserved) {
        if (r.name == "SALT") salt = &r;
    }
    if (!salt) fail(defPath, lineNo, "missing 'reserved SALT uint <index>'");
    std::uint64_t total = 1;
    for (const auto& a : axes) {
        total *= a.range();
        if (total > 0xFFFFFFu) fail(defPath, lineNo, "variant count overflows (> 16M)");
    }

    // ---- C++ header -------------------------------------------------------
    std::ostringstream h;
    h << "// GENERATED by tools/variant_gen from shaders/variants.def -- do not edit.\n"
         "#pragma once\n\n"
         "#include \"core/types.h\"\n"
         "#include \"pipeline/pipeline_desc.h\"\n\n"
         "#include <array>\n\n"
         "namespace phosphor::pipe::forward::gen {\n\n"
         "struct Axis {\n"
         "    const char*  name;\n"
         "    u16          constantIndex;\n"
         "    ConstantType type;\n"
         "    u32          minValue;\n"
         "    u32          maxValue;\n"
         "    u32          genericValue;     // value the generic shader assumes (0 when runtime)\n"
         "    bool         genericIsRuntime; // generic reads the value from a runtime buffer\n"
         "    constexpr u32 range() const { return maxValue - minValue + 1; }\n"
         "};\n\n"
         "inline constexpr u32 kAxisCount = " << axes.size() << ";\n"
         "inline constexpr u32 kVariantCount = " << total << ";\n"
         "inline constexpr u16 kSaltConstantIndex = " << salt->index << ";\n\n";
    for (std::size_t i = 0; i < axes.size(); ++i) {
        h << "inline constexpr u32 AXIS_" << axes[i].name << " = " << i << ";\n";
    }
    h << "\ninline constexpr std::array<Axis, kAxisCount> kAxes{{\n";
    for (const auto& a : axes) {
        h << "    {\"" << a.name << "\", " << a.index << ", ConstantType::" << (a.isBool ? "Bool" : "UInt") << ", "
          << a.minValue << ", " << a.maxValue << ", " << (a.runtime ? 0u : a.generic) << ", "
          << (a.runtime ? "true" : "false") << "},\n";
    }
    h << "}};\n\n"
         "using AxisValues = std::array<u32, kAxisCount>;\n\n"
         "/// Dense index of a value set (first axis = least significant digit).\n"
         "/// Values must lie inside each axis range.\n"
         "constexpr u32 indexOf(const AxisValues& values) {\n"
         "    u32 index = 0;\n"
         "    u32 stride = 1;\n"
         "    for (u32 i = 0; i < kAxisCount; ++i) {\n"
         "        index += (values[i] - kAxes[i].minValue) * stride;\n"
         "        stride *= kAxes[i].range();\n"
         "    }\n"
         "    return index;\n"
         "}\n\n"
         "/// Inverse of indexOf for index < kVariantCount.\n"
         "constexpr AxisValues valuesAt(u32 index) {\n"
         "    AxisValues values{};\n"
         "    for (u32 i = 0; i < kAxisCount; ++i) {\n"
         "        values[i] = kAxes[i].minValue + index % kAxes[i].range();\n"
         "        index /= kAxes[i].range();\n"
         "    }\n"
         "    return values;\n"
         "}\n\n"
         "} // namespace phosphor::pipe::forward::gen\n";

    // ---- MSL header -------------------------------------------------------
    std::ostringstream m;
    m << "// GENERATED by tools/variant_gen from shaders/variants.def -- do not edit.\n"
         "//\n"
         "// Function constants of the forward pass.  With no constant defined (the\n"
         "// generic pipeline) every k<Name> equals the generic value of its axis, or\n"
         "// -- for `runtime` axes -- the shader must consult k<Name>Specialised and read\n"
         "// the value from its runtime buffer.\n"
         "#pragma once\n\n";
    for (const auto& a : axes) {
        const std::string type = a.isBool ? "bool" : "uint";
        const std::string c = camel(a.name);
        m << "constant " << type << " FC_" << a.name << " [[function_constant(" << a.index << ")]];\n";
        m << "constant bool k" << c << "Specialised = is_function_constant_defined(FC_" << a.name << ");\n";
        std::string fallback;
        if (a.isBool) fallback = (a.generic ? "true" : "false");
        else fallback = std::to_string(a.runtime ? 0u : a.generic) + "u";
        m << "constant " << type << " k" << c << " = k" << c << "Specialised ? FC_" << a.name << " : " << fallback
          << ";" << (a.runtime ? " // generic: read the runtime value instead" : "") << "\n\n";
    }
    m << "constant uint FC_" << salt->name << " [[function_constant(" << salt->index << ")]];\n";

    const std::filesystem::path outDir = std::filesystem::path(argv[2]) / "pipeline";
    if (!writeIfChanged(outDir / "forward_variants.generated.h", h.str())) return 1;
    if (!writeIfChanged(outDir / "forward_variants.generated.metal.h", m.str())) return 1;
    return 0;
}
