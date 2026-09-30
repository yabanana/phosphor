// B-01: ALU throughput per type (FP32, FP16, INT32) and operation (FMA/MAD,
// ADD, MUL), N chains per thread (N = 1: one dependent chain = latency bound;
// N = 8: independent chains = throughput bound).  Operands come from a
// constant buffer so nothing folds at compile time; the result is stored.
#include <metal_stdlib>
using namespace metal;

struct AluParams {
    uint  iters;
    uint  pad;
    float xf, yf;   // FP32/FP16 operands
    int   xi, yi;   // INT32 operands
    uint  pad2, pad3;
};

template <typename T> inline T xOf(constant AluParams& p);
template <> inline float xOf<float>(constant AluParams& p) { return p.xf; }
template <> inline half  xOf<half>(constant AluParams& p) { return half(p.xf); }
template <> inline int   xOf<int>(constant AluParams& p) { return p.xi; }
template <typename T> inline T yOf(constant AluParams& p);
template <> inline float yOf<float>(constant AluParams& p) { return p.yf; }
template <> inline half  yOf<half>(constant AluParams& p) { return half(p.yf); }
template <> inline int   yOf<int>(constant AluParams& p) { return p.yi; }

// Different per thread and per chain: identical chains are merged by the
// compiler and thread-invariant ones run once per SIMD-group (measured).
template <typename T> inline T initOf(uint i, uint c) { return T(int((i + c * 7u) & 15u)) * T(0.0625) + T(1); }
template <> inline int initOf<int>(uint i, uint c) { return int(i * 2654435761u + c * 40503u + 1u); }

inline float mad_(float a, float x, float y) { return fma(a, x, y); }
inline half  mad_(half a, half x, half y) { return fma(a, x, y); }
inline int   mad_(int a, int x, int y) { return a * x + y; }

// ADD partner: float b - a (kept exactly: the library is compiled with
// MathModeSafe); int b ^ a (integer identities like b - (a + b) = -a apply
// even without fast math, xor has none).
inline float addPartner(float b, float a) { return b - a; }
inline half  addPartner(half b, half a) { return b - a; }
inline int   addPartner(int b, int a) { return b ^ a; }

inline uint bitsOf(float v) { return as_type<uint>(v); }
inline uint bitsOf(half v) { return uint(as_type<ushort>(v)); }
inline uint bitsOf(int v) { return uint(v); }

// OP 0 = FMA/MAD (a = a*x + y), 1 = ADD, 2 = MUL (a = a*x).
// ADD uses N pairs (a += b; then b -= a for floats, b ^= a for ints) = 2
// serial ops per pair and iteration: a single "a += x" is an affine
// induction that the compiler replaces by a + iters*x for integers.
template <typename T, int OP, int N>
kernel void alu(device uint* out [[buffer(0)]], constant AluParams& p [[buffer(1)]],
                uint i [[thread_position_in_grid]]) {
    const T x = xOf<T>(p), y = yOf<T>(p);
    T v[N], w[N];
    for (int c = 0; c < N; ++c) {
        v[c] = initOf<T>(i, uint(c)); // per thread and per chain: no uniform or merged chains
        w[c] = y + T(c);
    }
    for (uint k = 0; k < p.iters; ++k) {
        for (int c = 0; c < N; ++c) {
            if (OP == 0) v[c] = mad_(v[c], x, y);
            else if (OP == 1) { v[c] = v[c] + w[c]; w[c] = addPartner(w[c], v[c]); } // 2 serial ops
            else v[c] = v[c] * x;
        }
    }
    uint h = 0;
    for (int c = 0; c < N; ++c) h = h * 31u + bitsOf(v[c]) + (OP == 1 ? bitsOf(w[c]) * 17u : 0u);
    out[i] = h;
}

#define ALU_INST(T, TN, OP, ON, N) \
    template [[host_name("alu_" #TN "_" #ON "_" #N)]] kernel void alu<T, OP, N>(device uint*, constant AluParams&, uint);
#define ALU_TYPE(T, TN) \
    ALU_INST(T, TN, 0, fma, 1) ALU_INST(T, TN, 0, fma, 8) \
    ALU_INST(T, TN, 1, add, 1) ALU_INST(T, TN, 1, add, 8) \
    ALU_INST(T, TN, 2, mul, 1) ALU_INST(T, TN, 2, mul, 8)
ALU_TYPE(float, f32)
ALU_TYPE(half, f16)
ALU_TYPE(int, i32)
