// B-03: cost of transcendental / division operations.
//
// Float kernels: NCH independent chains per thread, each link is
//     v = op( fma(v, a, b) )
// with (a, b) per op chosen so that the map is a contraction whose fixed point
// is inside the domain of the op; the values therefore stay finite for any
// iteration count, the compiler cannot fold anything (the op is non-linear and
// its constants come from a buffer), and the CPU can verify the stored sum
// against the fixed point computed in double precision (the approximations of
// fast:: and half differ from bit-exact CPU code, so the check is a tolerance).
// The same kernel with op = identity ("glue") gives the cost of the fma alone
// (subtracted by the host).  Initial values differ per thread and per chain.
//
// Modes: 0 precise::, 1 fast::, 2 half (default namespace), 3 default calls
// (no namespace; the host compiles this file a second time with fast math on
// and uses the "def" kernels of that library).
//
// Integer kernels: x = x*K + C;  x ^= r(x)  with r a shift / mask / divide /
// modulo by a runtime or compile-time constant; bit-exact CPU check.
#include <metal_stdlib>
using namespace metal;

struct TrParams {
    uint  iters, pad;
    float a, b;     // glue fma constants
    float c1, c2;   // 1.0 (numerator), 0.75 (pow exponent)
    uint  d, pad2;  // runtime divisor
};

enum { RCP, RSQRT, SQRT, EXP2, LOG2, SIN, COS, POW, DIV, GLUE };
enum { PRECISE, FAST, HALF, DEF };

template <int OP, int MODE> struct Op;
#define DEF_OP(OPN, MODEN, TT, EXPR)                                  \
    template <> struct Op<OPN, MODEN> {                               \
        using T = TT;                                                 \
        static inline T f(T v, T c1, T c2) { return EXPR; }           \
    };
#define DEF_MODE(MODEN, TT, NS, ONE, RCPE, DIVE, POWE)                \
    DEF_OP(RCP, MODEN, TT, RCPE)                                      \
    DEF_OP(RSQRT, MODEN, TT, NS rsqrt(v))                             \
    DEF_OP(SQRT, MODEN, TT, NS sqrt(v))                               \
    DEF_OP(EXP2, MODEN, TT, NS exp2(v))                               \
    DEF_OP(LOG2, MODEN, TT, NS log2(v))                               \
    DEF_OP(SIN, MODEN, TT, NS sin(v))                                 \
    DEF_OP(COS, MODEN, TT, NS cos(v))                                 \
    DEF_OP(POW, MODEN, TT, POWE)                                      \
    DEF_OP(DIV, MODEN, TT, DIVE)                                      \
    DEF_OP(GLUE, MODEN, TT, v)

DEF_MODE(PRECISE, float, precise::, 1.0f, precise::divide(1.0f, v), precise::divide(c1, v), precise::pow(v, c2))
DEF_MODE(FAST, float, fast::, 1.0f, fast::divide(1.0f, v), fast::divide(c1, v), fast::pow(v, c2))
DEF_MODE(HALF, half, , 1.0h, half(1.0h) / v, c1 / v, pow(v, c2))
DEF_MODE(DEF, float, , 1.0f, 1.0f / v, c1 / v, pow(v, c2))

constant constexpr int NCH = 8;

template <int OP, int MODE>
kernel void tr(device uint* out [[buffer(0)]], constant TrParams& p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    using T = typename Op<OP, MODE>::T;
    const T a = T(p.a), b = T(p.b), c1 = T(p.c1), c2 = T(p.c2);
    T v[NCH];
    for (int c = 0; c < NCH; ++c) v[c] = T(0.5f + float((i + uint(c) * 7u) & 15u) * 0.125f);
    for (uint k = 0; k < p.iters; ++k)
        for (int c = 0; c < NCH; ++c) v[c] = Op<OP, MODE>::f(fma(v[c], a, b), c1, c2);
    float s = 0.0f;
    for (int c = 0; c < NCH; ++c) s += float(v[c]);
    out[i] = as_type<uint>(s);
}

#define INST(OPN, ONAME, MODEN, MNAME) \
    template [[host_name("tr_" #ONAME "_" #MNAME)]] kernel void tr<OPN, MODEN>(device uint*, constant TrParams&, uint);
#define INST_MODE(MODEN, MNAME)                                                                             \
    INST(RCP, rcp, MODEN, MNAME) INST(RSQRT, rsqrt, MODEN, MNAME) INST(SQRT, sqrt, MODEN, MNAME)            \
    INST(EXP2, exp2, MODEN, MNAME) INST(LOG2, log2, MODEN, MNAME) INST(SIN, sin, MODEN, MNAME)              \
    INST(COS, cos, MODEN, MNAME) INST(POW, pow, MODEN, MNAME) INST(DIV, div, MODEN, MNAME)                  \
    INST(GLUE, glue, MODEN, MNAME)
INST_MODE(PRECISE, precise)
INST_MODE(FAST, fast)
INST_MODE(HALF, half)
INST_MODE(DEF, def)

// --- Integer -------------------------------------------------------------------
enum { IBASE, ISHIFT, IMASK, IUDIV_RT, IUMOD_RT, IUDIV_C7, IUDIV_C8, ISDIV_RT, IUMOD_C7 };

template <int V> inline uint ir(uint x, uint d) {
    if (V == ISHIFT) return x >> 3u;
    if (V == IMASK) return x & 7u;
    if (V == IUDIV_RT) return x / d;
    if (V == IUMOD_RT) return x % d;
    if (V == IUDIV_C7) return x / 7u;
    if (V == IUDIV_C8) return x / 8u;
    if (V == ISDIV_RT) return uint(int(x) / int(d));
    if (V == IUMOD_C7) return x % 7u;
    return d; // base: xor with a runtime value (the glue: mad + xor)
}

template <int V>
kernel void iv(device uint* out [[buffer(0)]], constant TrParams& p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    uint x[NCH];
    for (int c = 0; c < NCH; ++c) x[c] = i * 2654435761u + uint(c) * 40503u + 1u;
    for (uint k = 0; k < p.iters; ++k)
        for (int c = 0; c < NCH; ++c) {
            x[c] = x[c] * 1664525u + 1013904223u;
            x[c] ^= ir<V>(x[c], p.d);
        }
    uint h = 0;
    for (int c = 0; c < NCH; ++c) h = h * 31u + x[c];
    out[i] = h;
}

#define IINST(V, NAME) template [[host_name("iv_" #NAME)]] kernel void iv<V>(device uint*, constant TrParams&, uint);
IINST(IBASE, base) IINST(ISHIFT, shift) IINST(IMASK, mask) IINST(IUDIV_RT, udiv_rt) IINST(IUMOD_RT, umod_rt)
IINST(IUDIV_C7, udiv_c7) IINST(IUDIV_C8, udiv_c8) IINST(ISDIV_RT, sdiv_rt) IINST(IUMOD_C7, umod_c7)
