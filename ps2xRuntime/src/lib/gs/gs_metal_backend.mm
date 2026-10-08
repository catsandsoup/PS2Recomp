// GSMetalBackend: Metal plan M1 skeleton. See gs_metal_backend.h for the design summary and
// game/docs/METAL_BACKEND_PLAN.md (section 13) for the measurements.
//
// Exactness notes (the oracle is gs_cpu_backend.cpp, Raster<NoTexels, C32, Z24>):
// - Clang compiles the oracle with -ffp-contract=on; the LLVM IR (checked with -emit-llvm) is
//     W  = ((fma(A, px - fx2, B * (py - fy2))) * winding) * invAbsDenom
//     w2 = (1 - w0) - w1
//     c  = fma(c2, w2, fma(c0, w0, c1 * w1))          (colour and fog, float)
//     z  = fma(z2, w2, fma(z0, w0, z1 * w1))          (double; the products are exact)
//   and the draw runs under the FPCR of the thread that issued it: round toward zero + FZ for the
//   game thread. The GPU rounds to nearest even, so the shader emulates RTZ with error-free
//   transformations (TwoSum / fma residuals) and Z with a 64-bit-integer soft double.
// - Per-primitive constants (edge coefficients, 1/|denom|, bounding box, sprite rectangle, sprite
//   Z) are computed on the CPU with the oracle's own expressions under the same FPCR.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "runtime/gs/gs_metal_backend.h"
#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/ps2_gs_common.h"
#include "runtime/gs/ps2_gs_memory.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

namespace
{
    // Shared with the shader (all 4-byte fields).
    struct GPUPrim
    {
        float fx2, fy2;
        float a0, b0, a1, b1;
        float winding, invAbsDenom;
        float z0, z1, z2;
        uint32_t rgba0, rgba1, rgba2;
        uint32_t fog;    // fog0 | fog1 << 8 | fog2 << 16
        uint32_t flags;  // 1 sprite, 2 iip, 4 rtz, 8 fge, 16 abe, 32 pabe, 64 fba, 128 zmask, 256 fault injection
        uint32_t test;   // TEST bits 0..31
        uint32_t alpha;  // A | B << 2 | C << 4 | D << 6 | FIX << 8
        uint32_t fbmsk;
        uint32_t fogcol; // r | g << 8 | b << 16
        uint32_t spriteZ;
        // textured triangles (G2a)
        float s0, t0, q0, s1, t1, q1, s2, t2, q2;
        uint32_t uv0, uv1, uv2; // u | v << 16 (raw 12.4)
        uint32_t tflags;        // 1 textured, 2 fst, 4 linear, 8 indexed, tfx << 4, tcc << 6, wms << 8, wmt << 10, 4096 raw target texture, 8192 CT24 (TEXA)
        uint32_t texdim;        // wrap size: w | h << 16
        uint32_t regU;          // minU | maxU << 16
        uint32_t regV;
        uint32_t palOff;        // word offset into the palette buffer (indexed textures)
        // G2b: sprites reuse s0,t0,s1,t1 = u0f,v0f,u1f,v1f; q0,q1 = sprite W,H; uv0,uv1 = unclipped x0,y0 (int bits)
        uint32_t texa;          // ta0 | aem << 8 (CT24 from a framebuffer target)
        uint32_t fbw;
    };
    static_assert(sizeof(GPUPrim) == 84 + 19 * 4, "GPUPrim layout");

    struct GPUVertex
    {
        float x, y;
        uint32_t prim;
    };

    const char *kShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Prim {
    float fx2, fy2;
    float a0, b0, a1, b1;
    float winding, invAbsDenom;
    float z0, z1, z2;
    uint rgba0, rgba1, rgba2;
    uint fog;
    uint flags;
    uint test;
    uint alpha;
    uint fbmsk;
    uint fogcol;
    uint spriteZ;
    float s0, t0, q0, s1, t1, q1, s2, t2, q2;
    uint uv0, uv1, uv2;
    uint tflags;
    uint texdim;
    uint regU;
    uint regV;
    uint palOff;
    uint texa;
    uint fbw;
};

struct Vtx { float x; float y; uint prim; };

struct VOut {
    float4 pos [[position]];
    uint prim [[flat]];
};

// ---------------------------------------------------------------- RTZ float emulation
// r is the round-to-nearest result, err has the sign of (exact - r).
inline float tz_fix(float r, float err)
{
    if (err == 0.0f || r == 0.0f || isinf(r))
        return r;
    if ((err < 0.0f) != (r < 0.0f))
        return as_type<float>(as_type<uint>(r) - 1u);
    return r;
}

inline float two_sum_err(float a, float b, float s)
{
    float bb = s - a;
    return (a - (s - bb)) + (b - bb);
}

// GPU arithmetic flushes denormals, so residuals of operations on small values (a product's residual
// can be 2^-47 of the product) are computed on operands scaled by 2^64 (exact, no overflow below 2^60).
constant float kTiny = 0x1p-60f;
constant float kBig = 0x1p60f;
constant float kScale = 0x1p64f;

inline float f_add(float a, float b, bool rtz)
{
    float s = a + b;
    if (!rtz)
        return s;
    const float lo = min(fabs(a), fabs(b));
    if (lo != 0.0f && lo < kTiny && max(fabs(a), fabs(b)) < kBig) {
        const float as = a * kScale, bs = b * kScale;
        return tz_fix(s, two_sum_err(as, bs, as + bs));
    }
    return tz_fix(s, two_sum_err(a, b, s));
}

inline float f_sub(float a, float b, bool rtz) { return f_add(a, -b, rtz); }

inline float f_mul(float a, float b, bool rtz)
{
    float p = a * b;
    if (!rtz)
        return p;
    if (fabs(p) < kTiny) {
        const float as = (fabs(a) < fabs(b)) ? a * kScale : a;
        const float bs = (fabs(a) < fabs(b)) ? b : b * kScale;
        const float ps = as * bs;
        return tz_fix(p, fma(as, bs, -ps));
    }
    return tz_fix(p, fma(a, b, -p));
}

// sign of (a*b + c - r), r = fma(a, b, c) rounded to nearest
inline float fma_residual(float a, float b, float c, float r)
{
    float p = a * b;
    float pe = fma(a, b, -p);       // a*b = p + pe exactly
    float s = p + c;
    float se = two_sum_err(p, c, s); // p + c = s + se exactly
    float d = s - r;                 // exact (s and r are close)
    float u = d + se;
    float ue = two_sum_err(d, se, u);
    float v = u + pe;
    return (v != 0.0f) ? v : ue;
}

inline float f_fma(float a, float b, float c, bool rtz)
{
    float r = fma(a, b, c);
    if (!rtz)
        return r;
    const float ap = fabs(a * b), ac = fabs(c);
    const float lo = (ap == 0.0f) ? ac : ((ac == 0.0f) ? ap : min(ap, ac));
    if (lo != 0.0f && lo < kTiny && max(max(ap, ac), fabs(r)) < kBig) {
        const bool sa = fabs(a) < fabs(b);
        return tz_fix(r, fma_residual(sa ? a * kScale : a, sa ? b : b * kScale, c * kScale, r * kScale));
    }
    return tz_fix(r, fma_residual(a, b, c, r));
}

// ---------------------------------------------------------------- soft double (value = +-m * 2^e)
struct SD { ulong m; int e; bool neg; };

inline SD sd_prod(float z, float w)
{
    uint zb = as_type<uint>(z), wb = as_type<uint>(w);
    uint ze = (zb >> 23) & 0xFFu, we = (wb >> 23) & 0xFFu;
    SD r;
    r.neg = ((zb ^ wb) >> 31) != 0u;
    if (ze == 0u || we == 0u) { r.m = 0ul; r.e = 0; return r; } // zero / flushed denormal
    ulong zm = ulong((zb & 0x7FFFFFu) | 0x800000u);
    ulong wm = ulong((wb & 0x7FFFFFu) | 0x800000u);
    r.m = zm * wm;
    r.e = int(ze) - 150 + int(we) - 150;
    return r;
}

inline SD sd_norm62(SD a)
{
    int sh = int(clz(a.m)) - 1;
    a.m <<= ulong(sh);
    a.e -= sh;
    return a;
}

// a + b rounded to a 53-bit significand (nearest-even, or toward zero).
inline SD sd_add(SD a, SD b, bool rtz)
{
    if (a.m == 0ul) return b;
    if (b.m == 0ul) return a;
    a = sd_norm62(a);
    b = sd_norm62(b);
    if (a.e < b.e || (a.e == b.e && a.m < b.m)) { SD t = a; a = b; b = t; }
    int d = a.e - b.e;
    ulong bm;
    bool sticky;
    if (d == 0) { bm = b.m; sticky = false; }
    else if (d < 64) { bm = b.m >> ulong(d); sticky = (b.m << ulong(64 - d)) != 0ul; }
    else { bm = 0ul; sticky = true; }
    ulong R;
    if (a.neg == b.neg) R = a.m + bm;
    else R = a.m - bm - (sticky ? 1ul : 0ul);
    SD r;
    r.neg = a.neg;
    r.e = a.e;
    if (R == 0ul && !sticky) { r.m = 0ul; r.e = 0; r.neg = false; return r; }
    int L = 64 - int(clz(R));
    if (L > 53) {
        int sh = L - 53;
        ulong rem = R & ((1ul << ulong(sh)) - 1ul);
        ulong halfv = 1ul << ulong(sh - 1);
        R >>= ulong(sh);
        r.e += sh;
        if (!rtz) {
            bool up = rem > halfv || (rem == halfv && (sticky || (R & 1ul) != 0ul));
            if (up) {
                R += 1ul;
                if (R == (1ul << 53)) { R >>= 1ul; r.e += 1; }
            }
        }
    }
    r.m = R;
    return r;
}

inline uint sd_to_u32(SD x)
{
    if (x.m == 0ul || x.neg) return 0u;
    if (x.e >= 0) {
        int L = 64 - int(clz(x.m));
        if (L + x.e > 32) return 0xFFFFFFFFu;
        return uint(x.m << ulong(x.e));
    }
    int s = -x.e;
    if (s >= 64) return 0u;
    ulong v = x.m >> ulong(s);
    return v > 0xFFFFFFFFul ? 0xFFFFFFFFu : uint(v);
}

inline uint tri_z(float z0, float z1, float z2, float w0, float w1, float w2, bool rtz)
{
    SD p0 = sd_prod(z0, w0), p1 = sd_prod(z1, w1), p2 = sd_prod(z2, w2);
    SD s = sd_add(p0, p1, rtz);
    s = sd_add(p2, s, rtz);
    SD h; h.m = 1ul; h.e = -1; h.neg = false;
    s = sd_add(s, h, rtz);
    return sd_to_u32(s);
}

inline uint interp_u8(uint c0, uint c1, uint c2, float w0, float w1, float w2, bool rtz)
{
    float v = f_fma(float(c2), w2, f_fma(float(c0), w0, f_mul(float(c1), w1, rtz), rtz), rtz);
    int i = int(v);
    return uint(clamp(i, 0, 255));
}

// ---------------------------------------------------------------- texture path (G2a)
// fcvtzs: float -> int32, truncate toward zero, saturating, NaN -> 0 (what the oracle's static_cast<int> compiles to).
inline int cvt_s32(float x)
{
    if (isnan(x)) return 0;
    if (x >= 2147483648.0f) return 0x7FFFFFFF;
    if (x <= -2147483648.0f) return int(0x80000000u);
    return int(x);
}

// sitofp under the draw's rounding mode (only differs from the nearest conversion above 2^24).
inline float i2f(int c, bool rtz)
{
    float f = float(c);
    if (rtz && (c > 16777216 || c < -16777216)) {
        long back = (fabs(f) >= 2147483648.0f) ? ((f < 0.0f) ? -2147483648L : 2147483648L) : long(f);
        long ex = long(c);
        if ((back < 0 ? -back : back) > (ex < 0 ? -ex : ex))
            f = as_type<float>(as_type<uint>(f) - 1u);
    }
    return f;
}

// 1.0f / b, correctly rounded in the draw's rounding mode (RTZ or RNE), flush-to-zero results.
// Metal's division is only accurate to a few ULP, so the quotient is corrected with exact fma residuals.
inline float f_rcp(float b, bool rtz)
{
    const float ab = fabs(b);
    float q;
    if (ab > 0x1p126f) {
        q = 0.0f;
    } else {
        q = 1.0f / ab;
        for (int i = 0; i < 4; ++i) {
            if (fma(-q, ab, 1.0f) < 0.0f) q = as_type<float>(as_type<uint>(q) - 1u); else break;
        }
        for (int i = 0; i < 4; ++i) {
            const float qn = as_type<float>(as_type<uint>(q) + 1u);
            if (fma(-qn, ab, 1.0f) >= 0.0f) q = qn; else break;
        }
        if (!rtz) {
            const float qn = as_type<float>(as_type<uint>(q) + 1u);
            const float r2 = 2.0f * fma(-q, ab, 1.0f);
            const float h = ab * (qn - q);
            if (r2 > h || (r2 == h && (as_type<uint>(q) & 1u) != 0u)) q = qn;
        }
    }
    return (b < 0.0f) ? -q : q;
}

// a / b for finite a > 0, b > 0, correctly rounded in the draw's rounding mode (the sprite axis division).
inline float f_div(float a, float b, bool rtz)
{
    float q = a / b;
    for (int i = 0; i < 4; ++i) {
        if (fma(-q, b, a) < 0.0f) q = as_type<float>(as_type<uint>(q) - 1u); else break;
    }
    for (int i = 0; i < 4; ++i) {
        const float qn = as_type<float>(as_type<uint>(q) + 1u);
        if (fma(-qn, b, a) >= 0.0f) q = qn; else break;
    }
    if (!rtz) {
        const float qn = as_type<float>(as_type<uint>(q) + 1u);
        const float r2 = 2.0f * fma(-q, b, a);
        const float h = b * (qn - q);
        if (r2 > h || (r2 == h && (as_type<uint>(q) & 1u) != 0u)) q = qn;
    }
    return q;
}

inline float interp3f(float a0, float a1, float a2, float w0, float w1, float w2, bool rtz)
{
    return f_fma(a2, w2, f_fma(a0, w0, f_mul(a1, w1, rtz), rtz), rtz);
}

// Sampler::Coord for the non-FST path: st * (1/fabsQ(q)) * size
inline float tex_coord_stq(float st, float q, uint size, bool rtz)
{
    const float aq = (fabs(q) > 1.0e-8f) ? q : 1.0f;
    return f_mul(f_mul(st, f_rcp(aq, rtz), rtz), float(size), rtz);
}

inline int wrap_coord(int c, uint size, uint mode, uint rmin, uint rmax)
{
    switch (mode & 3u) {
    case 0u: return int(uint(c) & (size - 1u));
    case 1u: return (c < 0) ? 0 : ((c > int(size) - 1) ? int(size) - 1 : c);
    case 2u: return min(max(c, int(rmin)), int(rmax));
    default: return int((uint(c) & rmin) | rmax);
    }
}

struct Axis { uint i0; uint i1; float frac; };

inline Axis prep_axis(float coord, uint size, uint mode, uint rmin, uint rmax, bool linear, bool rtz)
{
    Axis a;
    if (!linear) {
        a.i0 = uint(wrap_coord(cvt_s32(coord), size, mode, rmin, rmax));
        a.i1 = 0u;
        a.frac = 0.0f;
        return a;
    }
    const float sample = f_sub(coord, 0.5f, rtz);
    const int c0 = cvt_s32(floor(sample));
    a.frac = f_sub(sample, i2f(c0, rtz), rtz);
    a.i0 = uint(wrap_coord(c0, size, mode, rmin, rmax));
    a.i1 = uint(wrap_coord(int(uint(c0) + 1u), size, mode, rmin, rmax));
    return a;
}

inline uint fetch_texel(texture2d<uint, access::read> tex, const device uint *pal, uint palOff, uint tflags, uint texa, uint fbw, uint u, uint v)
{
    if ((tflags & 4096u) != 0u) {
        // a framebuffer target (raw 32-bit words): texel (u,v) is pixel (u,v); anything outside the target has
        // a bilinear weight the host proved negligible (see TryFbTexture)
        uint word = 0u;
        if (u < tex.get_width() && v < tex.get_height())
            word = tex.read(uint2(u, v)).x;
        if ((tflags & 8192u) != 0u) {
            const uint rgb = word & 0xFFFFFFu;
            const uint a = (((texa >> 8) & 1u) != 0u && rgb == 0u) ? 0u : (texa & 0xFFu);
            return rgb | (a << 24);
        }
        return word;
    }
    const uint4 t = tex.read(uint2(u, v));
    return ((tflags & 8u) != 0u) ? pal[palOff + (t.x & 0xFFu)] : (t.x | (t.y << 8) | (t.z << 16) | (t.w << 24));
}

// GSInternal::bilinearRgba8 (the NEON path: three fused multiply-adds per channel, round half away, saturate)
inline uint bilinear8(uint c00, uint c10, uint c01, uint c11, float fx, float fy, bool rtz)
{
    uint out = 0u;
    for (uint sh = 0u; sh < 32u; sh += 8u) {
        const float f00 = float((c00 >> sh) & 255u), f10 = float((c10 >> sh) & 255u);
        const float f01 = float((c01 >> sh) & 255u), f11 = float((c11 >> sh) & 255u);
        const float top = f_fma(f10 - f00, fx, f00, rtz);
        const float bottom = f_fma(f11 - f01, fx, f01, rtz);
        const float v = f_fma(f_sub(bottom, top, rtz), fy, top, rtz);
        const int i = cvt_s32(round(v));
        out |= uint(clamp(i, 0, 255)) << sh;
    }
    return out;
}

// ---------------------------------------------------------------- self-test kernels
kernel void selftest_f(device const float4 *in [[buffer(0)]], device uint4 *out [[buffer(1)]],
                       constant uint &rtz [[buffer(2)]], uint id [[thread_position_in_grid]])
{
    float4 v = in[id];
    bool r = rtz != 0u;
    out[id] = uint4(as_type<uint>(f_add(v.x, v.y, r)), as_type<uint>(f_mul(v.x, v.y, r)),
                    as_type<uint>(f_fma(v.x, v.y, v.z, r)), interp_u8(uint(v.w) & 255u, (uint(v.w) >> 8) & 255u, 7u, v.x, v.y, v.z, r));
}

kernel void selftest_z(device const float4 *zin [[buffer(0)]], device const float4 *win [[buffer(1)]],
                       device uint *out [[buffer(2)]], constant uint &rtz [[buffer(3)]], uint id [[thread_position_in_grid]])
{
    float4 z = zin[id];
    float4 w = win[id];
    out[id] = tri_z(z.x, z.y, z.z, w.x, w.y, w.z, rtz != 0u);
}

kernel void selftest_t(device const float4 *in [[buffer(0)]], device const uint4 *corners [[buffer(1)]],
                       device uint4 *out [[buffer(2)]], constant uint &rtz [[buffer(3)]], uint id [[thread_position_in_grid]])
{
    const float4 v = in[id];
    const uint4 c = corners[id];
    const bool r = rtz != 0u;
    out[id] = uint4(as_type<uint>(f_rcp(v.x, r)), as_type<uint>(tex_coord_stq(v.y, v.x, 256u, r)),
                    bilinear8(c.x, c.y, c.z, c.w, v.z, v.w, r), as_type<uint>(i2f(int(as_type<uint>(v.y)), r)));
}

// ---------------------------------------------------------------- draw
vertex VOut vs_main(uint vid [[vertex_id]], const device Vtx *v [[buffer(0)]], constant float2 &size [[buffer(1)]])
{
    VOut o;
    float2 p = float2(v[vid].x, v[vid].y);
    o.pos = float4(p.x / size.x * 2.0f - 1.0f, 1.0f - p.y / size.y * 2.0f, 0.0f, 1.0f);
    o.prim = v[vid].prim;
    return o;
}

struct FOut {
    uint frame [[color(0)]];
    uint depth [[color(1)]];
};

inline int pick_blend(uint sel, int cs, int cd) { return sel == 0u ? cs : (sel == 1u ? cd : 0); }

fragment FOut fs_main(VOut in [[stage_in]], const device Prim *prims [[buffer(0)]],
                      const device uint *pal [[buffer(1)]], texture2d<uint, access::read> tex [[texture(0)]],
                      uint fb [[color(0)]], uint zb [[color(1)]])
{
    const Prim P = prims[in.prim];
    const bool rtz = (P.flags & 4u) != 0u;
    uint r, g, b, a, fog, z;
    float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f;
    if ((P.flags & 1u) != 0u) {
        r = P.rgba0 & 0xFFu; g = (P.rgba0 >> 8) & 0xFFu; b = (P.rgba0 >> 16) & 0xFFu; a = P.rgba0 >> 24;
        fog = P.fog & 0xFFu;
        z = P.spriteZ;
    } else {
        const float px = in.pos.x, py = in.pos.y; // pixel centre = integer + 0.5
        const float dx = f_sub(px, P.fx2, rtz), dy = f_sub(py, P.fy2, rtz);
        w0 = f_mul(f_mul(f_fma(P.a0, dx, f_mul(P.b0, dy, rtz), rtz), P.winding, rtz), P.invAbsDenom, rtz);
        w1 = f_mul(f_mul(f_fma(P.a1, dx, f_mul(P.b1, dy, rtz), rtz), P.winding, rtz), P.invAbsDenom, rtz);
        w2 = f_sub(f_sub(1.0f, w0, rtz), w1, rtz);
        if (w0 < -1.0e-4f || w1 < -1.0e-4f || w2 < -1.0e-4f)
            discard_fragment();
        z = tri_z(P.z0, P.z1, P.z2, w0, w1, w2, rtz);
        if ((P.flags & 2u) != 0u) {
            r = interp_u8(P.rgba0 & 0xFFu, P.rgba1 & 0xFFu, P.rgba2 & 0xFFu, w0, w1, w2, rtz);
            g = interp_u8((P.rgba0 >> 8) & 0xFFu, (P.rgba1 >> 8) & 0xFFu, (P.rgba2 >> 8) & 0xFFu, w0, w1, w2, rtz);
            b = interp_u8((P.rgba0 >> 16) & 0xFFu, (P.rgba1 >> 16) & 0xFFu, (P.rgba2 >> 16) & 0xFFu, w0, w1, w2, rtz);
            a = interp_u8(P.rgba0 >> 24, P.rgba1 >> 24, P.rgba2 >> 24, w0, w1, w2, rtz);
        } else {
            r = P.rgba2 & 0xFFu; g = (P.rgba2 >> 8) & 0xFFu; b = (P.rgba2 >> 16) & 0xFFu; a = P.rgba2 >> 24;
        }
        fog = interp_u8(P.fog & 0xFFu, (P.fog >> 8) & 0xFFu, (P.fog >> 16) & 0xFFu, w0, w1, w2, rtz);

    }

    if ((P.tflags & 1u) != 0u) {
        {
            const bool fst = (P.tflags & 2u) != 0u, linear = (P.tflags & 4u) != 0u, indexed = (P.tflags & 8u) != 0u;
            const uint tfx = (P.tflags >> 4) & 3u, tcc = (P.tflags >> 6) & 1u;
            const uint wms = (P.tflags >> 8) & 3u, wmt = (P.tflags >> 10) & 3u;
            const uint texW = P.texdim & 0xFFFFu, texH = P.texdim >> 16;
            float cu, cv;
            if ((P.flags & 1u) != 0u) {
                // Raster::Sprite: per-axis position t = (x - x0 + 0.5) / W, coord = u0 + (u1 - u0) * t
                const int sx = int(in.pos.x), sy = int(in.pos.y);
                const float tx = f_div(float(sx - as_type<int>(P.uv0)) + 0.5f, P.q0, rtz);
                const float ty = f_div(float(sy - as_type<int>(P.uv1)) + 0.5f, P.q1, rtz);
                const float tu = f_fma(f_sub(P.s1, P.s0, rtz), tx, P.s0, rtz);
                const float tv = f_fma(f_sub(P.t1, P.t0, rtz), ty, P.t0, rtz);
                if (fst) {
                    cu = float(clamp(cvt_s32(f_fma(tu, 16.0f, 0.5f, rtz)), 0, 65535)) * 0.0625f;
                    cv = float(clamp(cvt_s32(f_fma(tv, 16.0f, 0.5f, rtz)), 0, 65535)) * 0.0625f;
                } else {
                    // texUf / texW * (1 / fabsQ(1)) * texW with power-of-two texW
                    const float rw = as_type<float>(0x3F800000u - (ctz(texW) << 23));
                    const float rh = as_type<float>(0x3F800000u - (ctz(texH) << 23));
                    cu = f_mul(f_mul(tu, rw, rtz), float(texW), rtz);
                    cv = f_mul(f_mul(tv, rh, rtz), float(texH), rtz);
                }
            } else if (fst) {
                const float fu = interp3f(float(P.uv0 & 0xFFFFu), float(P.uv1 & 0xFFFFu), float(P.uv2 & 0xFFFFu), w0, w1, w2, rtz);
                const float fv = interp3f(float(P.uv0 >> 16), float(P.uv1 >> 16), float(P.uv2 >> 16), w0, w1, w2, rtz);
                cu = float(uint(cvt_s32(fu)) & 0xFFFFu) * 0.0625f;
                cv = float(uint(cvt_s32(fv)) & 0xFFFFu) * 0.0625f;
            } else {
                const float is = interp3f(P.s0, P.s1, P.s2, w0, w1, w2, rtz);
                const float it = interp3f(P.t0, P.t1, P.t2, w0, w1, w2, rtz);
                const float iq = interp3f(P.q0, P.q1, P.q2, w0, w1, w2, rtz);
                cu = tex_coord_stq(is, iq, texW, rtz);
                cv = tex_coord_stq(it, iq, texH, rtz);
            }
            const Axis au = prep_axis(cu, texW, wms, P.regU & 0xFFFFu, P.regU >> 16, linear, rtz);
            const Axis av = prep_axis(cv, texH, wmt, P.regV & 0xFFFFu, P.regV >> 16, linear, rtz);
            uint texel;
            if (!linear) {
                texel = fetch_texel(tex, pal, P.palOff, P.tflags, P.texa, P.fbw, au.i0, av.i0);
            } else {
                const uint c00 = fetch_texel(tex, pal, P.palOff, P.tflags, P.texa, P.fbw, au.i0, av.i0);
                const uint c10 = fetch_texel(tex, pal, P.palOff, P.tflags, P.texa, P.fbw, au.i1, av.i0);
                const uint c01 = fetch_texel(tex, pal, P.palOff, P.tflags, P.texa, P.fbw, au.i0, av.i1);
                const uint c11 = fetch_texel(tex, pal, P.palOff, P.tflags, P.texa, P.fbw, au.i1, av.i1);
                texel = bilinear8(c00, c10, c01, c11, au.frac, av.frac, rtz);
            }
            const uint tr = texel & 0xFFu, tg = (texel >> 8) & 0xFFu, tb = (texel >> 16) & 0xFFu, ta = texel >> 24;
            const uint vr = r, vg = g, vb = b, va = a;
            uint nr = tr, ng = tg, nb = tb, na = (tcc != 0u) ? ta : va;
            switch (tfx) {
            case 0u: // MODULATE
                nr = min((tr * vr) >> 7, 255u); ng = min((tg * vg) >> 7, 255u); nb = min((tb * vb) >> 7, 255u);
                na = (tcc != 0u) ? min((ta * va) >> 7, 255u) : va;
                break;
            case 1u: // DECAL
                break;
            case 2u: // HIGHLIGHT
                nr = min(((tr * vr) >> 7) + va, 255u); ng = min(((tg * vg) >> 7) + va, 255u); nb = min(((tb * vb) >> 7) + va, 255u);
                na = (tcc != 0u) ? min(ta + va, 255u) : va;
                break;
            default: // HIGHLIGHT2
                nr = min(((tr * vr) >> 7) + va, 255u); ng = min(((tg * vg) >> 7) + va, 255u); nb = min(((tb * vb) >> 7) + va, 255u);
                na = (tcc != 0u) ? ta : va;
                break;
            }
            r = nr; g = ng; b = nb; a = na;
        }
    }

    // fog
    if ((P.flags & 8u) != 0u) {
        const uint inv = 255u - fog;
        r = ((fog * r) >> 8) + ((inv * (P.fogcol & 0xFFu)) >> 8);
        g = ((fog * g) >> 8) + ((inv * ((P.fogcol >> 8) & 0xFFu)) >> 8);
        b = ((fog * b) >> 8) + ((inv * ((P.fogcol >> 16) & 0xFFu)) >> 8);
        r &= 0xFFu; g &= 0xFFu; b &= 0xFFu;
    }

    // alpha test (CT32 frame)
    bool wRgb = true, wA = true, wZ = true;
    const uint test = P.test;
    if ((test & 1u) != 0u) {
        const uint atst = (test >> 1) & 7u, aref = (test >> 4) & 0xFFu;
        bool pass;
        switch (atst) {
        case 0u: pass = false; break;
        case 1u: pass = true; break;
        case 2u: pass = a < aref; break;
        case 3u: pass = a <= aref; break;
        case 4u: pass = a == aref; break;
        case 5u: pass = a >= aref; break;
        case 6u: pass = a > aref; break;
        default: pass = a != aref; break;
        }
        if (!pass) {
            switch ((test >> 12) & 3u) {
            case 1u: wRgb = true; wA = true; wZ = false; break;  // FB_ONLY
            case 2u: wRgb = false; wA = false; wZ = true; break; // ZB_ONLY
            case 3u: wRgb = true; wA = false; wZ = false; break; // RGB_ONLY (CT32)
            default: wRgb = false; wA = false; wZ = false; break; // KEEP
            }
        }
    }
    if (!wRgb && !wA && !wZ)
        discard_fragment();

    // destination alpha test
    if (((test >> 14) & 1u) != 0u) {
        const uint datm = (test >> 15) & 1u;
        if (((fb >> 31) & 1u) != datm)
            discard_fragment();
    }

    // depth test on the raw Z24 value
    const uint zcur = zb & 0xFFFFFFu;
    const uint ztst = (test >> 17) & 3u;
    bool zpass = (ztst == 1u) || (ztst == 2u && z >= zcur) || (ztst == 3u && z > zcur);
    if (!zpass)
        discard_fragment();

    FOut o;
    o.frame = fb;
    o.depth = zb;
    const bool writesFb = wRgb || wA;
    if (writesFb) {
        const bool preserveDestAlpha = wRgb && !wA;
        if ((P.flags & 16u) != 0u && !((P.flags & 32u) != 0u && (a & 0x80u) == 0u)) {
            const int dr = int(fb & 0xFFu), dg = int((fb >> 8) & 0xFFu), db = int((fb >> 16) & 0xFFu), da = int(fb >> 24);
            const uint asel = P.alpha & 3u, bsel = (P.alpha >> 2) & 3u, csel = (P.alpha >> 4) & 3u, dsel = (P.alpha >> 6) & 3u;
            const int fix = int((P.alpha >> 8) & 0xFFu);
            const int ca = csel == 0u ? int(a) : (csel == 1u ? da : fix);
            r = uint(clamp(((pick_blend(asel, int(r), dr) - pick_blend(bsel, int(r), dr)) * ca >> 7) + pick_blend(dsel, int(r), dr), 0, 255));
            g = uint(clamp(((pick_blend(asel, int(g), dg) - pick_blend(bsel, int(g), dg)) * ca >> 7) + pick_blend(dsel, int(g), dg), 0, 255));
            b = uint(clamp(((pick_blend(asel, int(b), db) - pick_blend(bsel, int(b), db)) * ca >> 7) + pick_blend(dsel, int(b), db), 0, 255));
        }
        if (wA && (P.flags & 64u) != 0u)
            a |= 0x80u;
        uint pixel = r | (g << 8) | (b << 16) | ((a & 0xFFu) << 24);
        if (P.fbmsk != 0u)
            pixel = (pixel & ~P.fbmsk) | (fb & P.fbmsk);
        if (preserveDestAlpha)
            pixel = (pixel & 0x00FFFFFFu) | (fb & 0xFF000000u);
        o.frame = pixel;
        if ((P.flags & 256u) != 0u)
            o.frame ^= 1u; // PS2X_GS_METAL_FAULT: deliberate error to prove the tee comparison sees it
    }
    if (wZ && (P.flags & 128u) == 0u)
        o.depth = (zb & 0xFF000000u) | (z & 0x00FFFFFFu);
    return o;
}
)METAL";

    constexpr uint32_t kPages = 512u;

    struct PageRange
    {
        uint32_t pages[512];
        uint32_t count = 0;
    };

    // Pages covered by a pixel rectangle of a 32-bit (64x32-page) or other PSM surface.
    void pagesOfRect(uint32_t blockBase, uint32_t bw, uint8_t psm, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, PageRange &out)
    {
        out.count = 0;
        uint32_t pw = 64u, ph = 32u;
        switch (psm)
        {
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            ph = 64u;
            break;
        case GS_PSM_T8:
            pw = 128u;
            ph = 64u;
            break;
        case GS_PSM_T4:
            pw = 128u;
            ph = 128u;
            break;
        default:
            break;
        }
        const uint32_t ppr = std::max<uint32_t>(1u, (std::max<uint32_t>(bw, 1u) * 64u) / pw);
        const uint32_t base = blockBase >> 5;
        const uint32_t extra = (blockBase & 31u) != 0u ? 1u : 0u;
        for (uint32_t py = y0 / ph; py <= y1 / ph; ++py)
            for (uint32_t px = x0 / pw; px <= x1 / pw + extra; ++px)
            {
                if (out.count >= 512u)
                    return;
                out.pages[out.count++] = (base + py * ppr + px) % kPages;
            }
    }

    uint64_t nowNs()
    {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
    }

    uint64_t readFpcr()
    {
#if defined(__aarch64__)
        uint64_t fpcr;
        __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
        return fpcr;
#else
        return 0;
#endif
    }

    bool fpcrIsTowardZero(uint64_t fpcr)
    {
#if defined(__aarch64__)
        return ((fpcr >> 22) & 3u) == 3u; // RMode = RZ
#else
        (void)fpcr;
        return std::fegetround() == FE_TOWARDZERO;
#endif
    }
}

void GSMetalBackend::Stats::Add(const Stats &o)
{
    submits += o.submits;
    metalPrims += o.metalPrims;
    metalTexPrims += o.metalTexPrims;
    metalFbPrims += o.metalFbPrims;
    metalSelfPrims += o.metalSelfPrims;
    fallbackPrims += o.fallbackPrims;
    runs += o.runs;
    targetUploads += o.targetUploads;
    readbackPixels += o.readbackPixels;
    gpuWaitNs += o.gpuWaitNs;
    presents += o.presents;
    texDecodes += o.texDecodes;
    texHits += o.texHits;
    texTexels += o.texTexels;
    palettes += o.palettes;
    paletteHits += o.paletteHits;
    for (uint32_t i = 0; i < kFlushWhyCount; ++i)
        flushes[i] += o.flushes[i];
    recordNs += o.recordNs;
    texNs += o.texNs;
    uploadNs += o.uploadNs;
    encodeNs += o.encodeNs;
    writebackNs += o.writebackNs;
    fallbackNs += o.fallbackNs;
    for (const auto &[k, v] : o.fallbacks)
        fallbacks[k] += v;
}

void GSMetalPrintStats(const char *tag, const GSMetalBackend::Stats &s)
{
    std::fprintf(stderr, "[gsmtl] %s submits=%llu metal=%llu (textured=%llu) fallback=%llu metal_frac=%.4f runs=%llu uploads=%llu readback_px=%llu gpu_wait_ms=%.1f presents=%llu\n",
                 tag, (unsigned long long)s.submits, (unsigned long long)s.metalPrims, (unsigned long long)s.metalTexPrims, (unsigned long long)s.fallbackPrims,
                 s.submits ? double(s.metalPrims) / double(s.submits) : 0.0, (unsigned long long)s.runs,
                 (unsigned long long)s.targetUploads, (unsigned long long)s.readbackPixels, double(s.gpuWaitNs) / 1e6, (unsigned long long)s.presents);
    std::fprintf(stderr, "[gsmtl] %s textures decodes=%llu hits=%llu texels=%llu palettes=%llu palette_hits=%llu fb_target_sprites=%llu self_snapshot_sprites=%llu\n", tag, (unsigned long long)s.texDecodes,
                 (unsigned long long)s.texHits, (unsigned long long)s.texTexels, (unsigned long long)s.palettes, (unsigned long long)s.paletteHits,
                 (unsigned long long)s.metalFbPrims, (unsigned long long)s.metalSelfPrims);
    static const char *const kWhy[GSMetalBackend::kFlushWhyCount] = {"target", "tex_overlap", "clut", "transfer", "fallback", "present", "readback", "other"};
    std::fprintf(stderr, "[gsmtl] %s run_closes", tag);
    for (uint32_t i = 0; i < GSMetalBackend::kFlushWhyCount; ++i)
        std::fprintf(stderr, " %s=%llu", kWhy[i], (unsigned long long)s.flushes[i]);
    std::fprintf(stderr, "\n");
    if (s.recordNs || s.texNs || s.uploadNs || s.encodeNs || s.writebackNs || s.fallbackNs)
        std::fprintf(stderr, "[gsmtl] %s time_ms record=%.1f tex_decode=%.1f target_upload=%.1f encode=%.1f gpu_wait=%.1f writeback=%.1f cpu_fallback=%.1f\n", tag,
                     double(s.recordNs) / 1e6, double(s.texNs) / 1e6, double(s.uploadNs) / 1e6, double(s.encodeNs) / 1e6, double(s.gpuWaitNs) / 1e6,
                     double(s.writebackNs) / 1e6, double(s.fallbackNs) / 1e6);
    for (const auto &[k, v] : s.fallbacks)
        std::fprintf(stderr, "[gsmtl] %s fallback %s=%llu\n", tag, k.c_str(), (unsigned long long)v);
    std::fflush(stderr);
}

struct GSMetalBackend::Impl
{
    struct Target
    {
        uint32_t fbp = 0, zbp = 0, fbw = 0;
        uint32_t width = 0, height = 0;
        id<MTLTexture> color = nil;
        id<MTLTexture> depth = nil;
        id<MTLBuffer> staging = nil; // 2 * width * height words (colour then depth)
        uint64_t syncEpoch = 0;
        bool valid = false;
    };

    // ---- textured triangles (G2a)
    struct TexKey
    {
        uint32_t tbp = 0, tbw = 0, psm = 0, w = 0, h = 0, texa = 0;
        bool operator==(const TexKey &o) const
        {
            return tbp == o.tbp && tbw == o.tbw && psm == o.psm && w == o.w && h == o.h && texa == o.texa;
        }
    };
    struct TexKeyHash
    {
        size_t operator()(const TexKey &k) const
        {
            uint64_t h = 0x9E3779B97F4A7C15ull;
            for (uint32_t v : {k.tbp, k.tbw, k.psm, k.w, k.h, k.texa})
            {
                h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
                h *= 0xFF51AFD7ED558CCDull;
            }
            return static_cast<size_t>(h ^ (h >> 29));
        }
    };
    struct TexEntry
    {
        id<MTLTexture> tex = nil;
        std::vector<uint16_t> pages; // VRAM pages the texels can come from
        uint64_t epoch = 0;          // global epoch at decode time
        uint64_t checked = 0;        // global epoch at the last validation
        uint64_t lastUse = 0;
        size_t bytes = 0;
    };
    struct TexSetup
    {
        bool on = false;
        TexKey key;
        uint32_t texW = 1, texH = 1;
        uint32_t wms = 0, wmt = 0, minU = 0, maxU = 0, minV = 0, maxV = 0;
        uint32_t tfx = 0, tcc = 0;
        bool indexed = false, fst = false, linear = false;
    };
    struct Segment
    {
        uint32_t first = 0;
        id<MTLTexture> tex = nil;
    };

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLRenderPipelineState> pipeline = nil;
    id<MTLComputePipelineState> selftestF = nil;
    id<MTLComputePipelineState> selftestZ = nil;
    id<MTLComputePipelineState> selftestT = nil;

    std::unique_ptr<GSCpuBackend> cpu;
    uint8_t *vram = nullptr;
    uint32_t vramSize = 0;
    bool timing = false;
    bool flushAll = false; // PS2X_GS_METAL_FLUSH_ALL=1: close the run before every CPU-side operation (M1 behaviour)

    std::array<uint64_t, kPages> pageEpoch{};
    uint64_t epoch = 1;
    std::vector<std::unique_ptr<Target>> targets;

    // open run
    Target *run = nullptr;
    std::vector<GPUVertex> verts;
    std::vector<GPUPrim> prims;
    std::vector<Segment> segments;
    id<MTLTexture> curTex = nil;
    std::bitset<kPages> runPages; // pages of the run target's colour and Z planes (conservative)
    int bx0 = 0, by0 = 0, bx1 = -1, by1 = -1;
    uint32_t runPrims = 0;

    // texture cache, palette buffer
    TexSetup ts;
    std::unordered_map<TexKey, TexEntry, TexKeyHash> texCache;
    size_t texBytes = 0;
    uint64_t useTick = 0;
    id<MTLTexture> dummyTex = nil;
    std::vector<uint8_t> scratch8;
    std::vector<uint32_t> scratch32;
    static constexpr uint32_t kPalSlots = 4096u;
    id<MTLBuffer> palBuf = nil;
    uint32_t palUsed = 0;
    std::unordered_map<uint64_t, uint32_t> palMap;
    bool lastPalValid = false;
    uint64_t lastPalGen = 0, lastPalKey = 0;
    uint32_t lastPalOff = 0;
    uint32_t curPalOff = 0;
    id<MTLTexture> primTex = nil;
    // textured sprites (G2b)
    id<MTLTexture> snapTex = nil;  // read-before-write copy of the run target for a feedback sprite
    Target *snapTarget = nullptr;
    bool fbSrc = false, fbCt24 = false;

    // transfer page tracking
    GSTransferCommand transfer{};
    PageRange transferPages{};

    std::function<void(const RunInfo &)> observer;
    Stats frame;
    Stats total;

    void MarkPages(const PageRange &r)
    {
        ++epoch;
        for (uint32_t i = 0; i < r.count; ++i)
            pageEpoch[r.pages[i]] = epoch;
    }

    void MarkAll()
    {
        ++epoch;
        pageEpoch.fill(epoch);
    }

    void TargetPages(const Target &t, bool depth, PageRange &out) const
    {
        pagesOfRect((depth ? t.zbp : t.fbp) << 5, t.fbw, GS_PSM_CT32, 0u, 0u, t.width - 1u, t.height - 1u, out);
    }

    bool TargetStale(const Target &t) const
    {
        if (!t.valid)
            return true;
        PageRange r;
        for (int d = 0; d < 2; ++d)
        {
            TargetPages(t, d != 0, r);
            for (uint32_t i = 0; i < r.count; ++i)
                if (pageEpoch[r.pages[i]] > t.syncEpoch)
                    return true;
        }
        return false;
    }

    void ComputeRunPages()
    {
        runPages.reset();
        if (!run)
            return;
        PageRange r;
        for (int d = 0; d < 2; ++d)
        {
            TargetPages(*run, d != 0, r);
            for (uint32_t i = 0; i < r.count; ++i)
                runPages.set(r.pages[i]);
        }
    }

    // Does an operation touching these pages need the open run's results in the shadow (or write
    // pages the run renders to)? A range that hit the 512-entry cap is treated as "everything".
    bool OverlapsRun(const PageRange &r) const
    {
        if (!run)
            return false;
        if (flushAll || r.count >= kPages)
            return true;
        for (uint32_t i = 0; i < r.count; ++i)
            if (runPages.test(r.pages[i]))
                return true;
        return false;
    }

    void FlushIfOverlaps(const PageRange &r, FlushWhy why)
    {
        if (OverlapsRun(r))
            FlushRun(why);
    }

    Target *GetTarget(uint32_t fbp, uint32_t zbp, uint32_t fbw, uint32_t needHeight)
    {
        const uint32_t height = std::max<uint32_t>(32u, (needHeight + 31u) & ~31u);
        for (auto &t : targets)
        {
            if (t->fbp == fbp && t->zbp == zbp && t->fbw == fbw)
            {
                if (t->height >= height)
                    return t.get();
                Allocate(*t, fbw * 64u, height);
                return t.get();
            }
        }
        auto t = std::make_unique<Target>();
        t->fbp = fbp;
        t->zbp = zbp;
        t->fbw = fbw;
        Allocate(*t, fbw * 64u, height);
        targets.push_back(std::move(t));
        return targets.back().get();
    }

    // The run's target for this draw (closing the open run when it is another target or too short).
    Target *SelectRun(const GSContext &ctx)
    {
        const uint32_t needHeight = uint32_t(ctx.scissor.y1) + 1u;
        if (run && run->fbp == ctx.frame.fbp && run->zbp == ctx.zbuf.zbp && run->fbw == ctx.frame.fbw && run->height >= needHeight)
            return run;
        FlushRun(kFlushTarget);
        Target *t = GetTarget(ctx.frame.fbp, ctx.zbuf.zbp, ctx.frame.fbw, needHeight);
        run = t;
        ComputeRunPages();
        return t;
    }

    void Allocate(Target &t, uint32_t width, uint32_t height)
    {
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
                                                                                     width:width
                                                                                    height:height
                                                                                 mipmapped:NO];
        d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        d.storageMode = MTLStorageModePrivate;
        t.color = [device newTextureWithDescriptor:d];
        t.depth = [device newTextureWithDescriptor:d];
        t.staging = [device newBufferWithLength:size_t(width) * height * 8u options:MTLResourceStorageModeShared];
        t.width = width;
        t.height = height;
        t.valid = false;
    }

    // Copy the target's pixels out of the shadow into the staging buffer (raw 32-bit words).
    void FillStaging(Target &t)
    {
        uint32_t *dst = static_cast<uint32_t *>(t.staging.contents);
        const GSMem::SwizzledSurface<GSMem::C32> cs(t.fbp << 5, t.fbw);
        const GSMem::SwizzledSurface<GSMem::Z24> zs(t.zbp << 5, t.fbw);
        uint32_t *dz = dst + size_t(t.width) * t.height;
        for (uint32_t y = 0; y < t.height; ++y)
            for (uint32_t x = 0; x < t.width; ++x)
            {
                uint32_t v;
                std::memcpy(&v, vram + cs.Locate(x, y).byteAddress, 4);
                dst[size_t(y) * t.width + x] = v;
                std::memcpy(&v, vram + zs.Locate(x, y).byteAddress, 4);
                dz[size_t(y) * t.width + x] = v;
            }
    }

    void FlushRun(FlushWhy why = kFlushOther)
    {
        if (!run)
            return;
        Target &t = *run;
        @autoreleasepool
        {
            const bool stale = TargetStale(t);
            if (!prims.empty() || stale)
            {
                ++frame.flushes[why];
                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                const size_t plane = size_t(t.width) * t.height * 4u;
                if (stale)
                {
                    const uint64_t tu0 = timing ? nowNs() : 0;
                    FillStaging(t);
                    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
                    [blit copyFromBuffer:t.staging sourceOffset:0 sourceBytesPerRow:t.width * 4u sourceBytesPerImage:plane
                              sourceSize:MTLSizeMake(t.width, t.height, 1) toTexture:t.color destinationSlice:0 destinationLevel:0
                       destinationOrigin:MTLOriginMake(0, 0, 0)];
                    [blit copyFromBuffer:t.staging sourceOffset:plane sourceBytesPerRow:t.width * 4u sourceBytesPerImage:plane
                              sourceSize:MTLSizeMake(t.width, t.height, 1) toTexture:t.depth destinationSlice:0 destinationLevel:0
                       destinationOrigin:MTLOriginMake(0, 0, 0)];
                    [blit endEncoding];
                    ++frame.targetUploads;
                    t.valid = true;
                    t.syncEpoch = epoch;
                    if (timing)
                        frame.uploadNs += nowNs() - tu0;
                }
                const bool draw = !prims.empty() && bx1 >= bx0 && by1 >= by0;
                if (draw)
                {
                    const uint64_t te0 = timing ? nowNs() : 0;
                    if (snapTex && snapTarget == &t)
                    {
                        id<MTLBlitCommandEncoder> sb = [cmd blitCommandEncoder];
                        [sb copyFromTexture:t.color sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                                 sourceSize:MTLSizeMake(t.width, t.height, 1) toTexture:snapTex destinationSlice:0 destinationLevel:0
                          destinationOrigin:MTLOriginMake(0, 0, 0)];
                        [sb endEncoding];
                    }
                    id<MTLBuffer> vb = [device newBufferWithBytes:verts.data() length:verts.size() * sizeof(GPUVertex) options:MTLResourceStorageModeShared];
                    id<MTLBuffer> pb = [device newBufferWithBytes:prims.data() length:prims.size() * sizeof(GPUPrim) options:MTLResourceStorageModeShared];
                    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
                    rp.colorAttachments[0].texture = t.color;
                    rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
                    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
                    rp.colorAttachments[1].texture = t.depth;
                    rp.colorAttachments[1].loadAction = MTLLoadActionLoad;
                    rp.colorAttachments[1].storeAction = MTLStoreActionStore;
                    id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:rp];
                    [enc setRenderPipelineState:pipeline];
                    MTLViewport vp = {0.0, 0.0, double(t.width), double(t.height), 0.0, 1.0};
                    [enc setViewport:vp];
                    const float size[2] = {float(t.width), float(t.height)};
                    [enc setVertexBuffer:vb offset:0 atIndex:0];
                    [enc setVertexBytes:size length:sizeof(size) atIndex:1];
                    [enc setFragmentBuffer:pb offset:0 atIndex:0];
                    [enc setFragmentBuffer:palBuf offset:0 atIndex:1];
                    if (segments.empty())
                    {
                        [enc setFragmentTexture:dummyTex atIndex:0];
                        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:verts.size()];
                    }
                    else
                    {
                        for (size_t i = 0; i < segments.size(); ++i)
                        {
                            const uint32_t first = segments[i].first;
                            const uint32_t last = (i + 1 < segments.size()) ? segments[i + 1].first : uint32_t(verts.size());
                            if (last <= first)
                                continue;
                            [enc setFragmentTexture:segments[i].tex atIndex:0];
                            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:first vertexCount:last - first];
                        }
                    }
                    [enc endEncoding];

                    // dirty rectangle back into the staging buffer (colour plane, then depth plane)
                    const uint32_t w = uint32_t(bx1 - bx0 + 1), h = uint32_t(by1 - by0 + 1);
                    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
                    [blit copyFromTexture:t.color sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(bx0, by0, 0)
                               sourceSize:MTLSizeMake(w, h, 1) toBuffer:t.staging destinationOffset:0
                      destinationBytesPerRow:w * 4u destinationBytesPerImage:size_t(w) * h * 4u];
                    [blit copyFromTexture:t.depth sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(bx0, by0, 0)
                               sourceSize:MTLSizeMake(w, h, 1) toBuffer:t.staging destinationOffset:plane
                      destinationBytesPerRow:w * 4u destinationBytesPerImage:size_t(w) * h * 4u];
                    [blit endEncoding];
                    if (timing)
                        frame.encodeNs += nowNs() - te0;
                }
                const uint64_t t0 = nowNs();
                [cmd commit];
                [cmd waitUntilCompleted];
                frame.gpuWaitNs += nowNs() - t0;
                if (cmd.status == MTLCommandBufferStatusError)
                {
                    std::fprintf(stderr, "[gsmtl] command buffer error: %s\n", cmd.error.localizedDescription.UTF8String);
                    std::fflush(stderr);
                }
                if (draw)
                {
                    const uint64_t tw0 = timing ? nowNs() : 0;
                    const uint32_t w = uint32_t(bx1 - bx0 + 1), h = uint32_t(by1 - by0 + 1);
                    const uint32_t *src = static_cast<const uint32_t *>(t.staging.contents);
                    const uint32_t *srcZ = reinterpret_cast<const uint32_t *>(static_cast<const uint8_t *>(t.staging.contents) + plane);
                    const GSMem::SwizzledSurface<GSMem::C32> cs(t.fbp << 5, t.fbw);
                    const GSMem::SwizzledSurface<GSMem::Z24> zs(t.zbp << 5, t.fbw);
                    for (uint32_t y = 0; y < h; ++y)
                        for (uint32_t x = 0; x < w; ++x)
                        {
                            std::memcpy(vram + cs.Locate(uint32_t(bx0) + x, uint32_t(by0) + y).byteAddress, &src[size_t(y) * w + x], 4);
                            std::memcpy(vram + zs.Locate(uint32_t(bx0) + x, uint32_t(by0) + y).byteAddress, &srcZ[size_t(y) * w + x], 4);
                        }
                    frame.readbackPixels += uint64_t(w) * h;
                    PageRange r;
                    pagesOfRect(t.fbp << 5, t.fbw, GS_PSM_CT32, uint32_t(bx0), uint32_t(by0), uint32_t(bx1), uint32_t(by1), r);
                    MarkPages(r);
                    pagesOfRect(t.zbp << 5, t.fbw, GS_PSM_Z24, uint32_t(bx0), uint32_t(by0), uint32_t(bx1), uint32_t(by1), r);
                    MarkPages(r);
                    t.syncEpoch = epoch; // the target equals the shadow again
                    ++frame.runs;
                    if (timing)
                        frame.writebackNs += nowNs() - tw0;
                }
            }
        }
        RunInfo info;
        info.fbp = t.fbp;
        info.zbp = t.zbp;
        info.fbw = t.fbw;
        info.x0 = bx0;
        info.y0 = by0;
        info.x1 = bx1;
        info.y1 = by1;
        info.prims = runPrims;
        const bool drew = !prims.empty() && bx1 >= bx0 && by1 >= by0;
        run = nullptr;
        snapTex = nil;
        snapTarget = nullptr;
        verts.clear();
        prims.clear();
        segments.clear();
        curTex = nil;
        runPages.reset();
        palUsed = 0;
        palMap.clear();
        lastPalValid = false;
        bx0 = by0 = 0;
        bx1 = by1 = -1;
        runPrims = 0;
        if (drew && observer)
            observer(info);
    }

    // CPU-side writes: mark pages the operation can touch.
    void MarkCpuDraw(const GSPrimitiveBatch &batch)
    {
        const GSContext &ctx = batch.state.context;
        PageRange r;
        const uint32_t sx1 = std::min<uint32_t>(ctx.scissor.x1, 2047u), sy1 = std::min<uint32_t>(ctx.scissor.y1, 2047u);
        pagesOfRect(ctx.frame.fbp << 5, ctx.frame.fbw, ctx.frame.psm, ctx.scissor.x0, ctx.scissor.y0, std::max<uint32_t>(sx1, ctx.scissor.x0),
                    std::max<uint32_t>(sy1, ctx.scissor.y0), r);
        MarkPages(r);
        pagesOfRect(ctx.zbuf.zbp << 5, ctx.frame.fbw, ctx.zbuf.psm, ctx.scissor.x0, ctx.scissor.y0, std::max<uint32_t>(sx1, ctx.scissor.x0),
                    std::max<uint32_t>(sy1, ctx.scissor.y0), r);
        MarkPages(r);
    }

    // ---------------------------------------------------------------- textured triangles
    // Fills `ts` for a textured triangle; returns a fallback reason or nullptr.
    const char *ParseTexture(const GSPrimitiveBatch &batch)
    {
        const GSDrawState &st = batch.state;
        const GSTex0Reg &tex = st.context.tex0;
        ts = TexSetup{};
        if (tex.psm != GS_PSM_T8 && tex.psm != GS_PSM_T4 && tex.psm != GS_PSM_CT32 && tex.psm != GS_PSM_CT24)
            return "tex_psm";
        ts.texW = st.textureWidth;
        ts.texH = st.textureHeight;
        if (ts.texW < 1u || ts.texW > 1024u || ts.texH < 1u || ts.texH > 1024u || (ts.texW & (ts.texW - 1u)) != 0u || (ts.texH & (ts.texH - 1u)) != 0u)
            return "tex_size";
        const uint64_t clamp = st.context.clamp;
        ts.wms = uint32_t(clamp & 3u);
        ts.wmt = uint32_t((clamp >> 2) & 3u);
        ts.minU = uint32_t((clamp >> 4) & 0x3FFu);
        ts.maxU = uint32_t((clamp >> 14) & 0x3FFu);
        ts.minV = uint32_t((clamp >> 24) & 0x3FFu);
        ts.maxV = uint32_t((clamp >> 34) & 0x3FFu);
        auto bound = [](uint32_t mode, uint32_t size, uint32_t rmin, uint32_t rmax) -> uint32_t
        {
            switch (mode & 3u)
            {
            case 0u:
            case 1u:
                return size - 1u;
            case 2u:
                return rmax;
            default:
                return rmin | rmax;
            }
        };
        ts.key.w = std::min<uint32_t>(bound(ts.wms, ts.texW, ts.minU, ts.maxU) + 1u, 1024u);
        ts.key.h = std::min<uint32_t>(bound(ts.wmt, ts.texH, ts.minV, ts.maxV) + 1u, 1024u);
        ts.key.tbp = tex.tbp0;
        ts.key.tbw = tex.tbw;
        ts.key.psm = tex.psm;
        ts.key.texa = (tex.psm == GS_PSM_CT24) ? (uint32_t(st.texa.ta0) | (st.texa.aem ? 0x100u : 0u) | 0x10000u) : 0u;
        ts.tfx = tex.tfx & 3u;
        ts.tcc = tex.tcc & 1u;
        ts.indexed = (tex.psm == GS_PSM_T8 || tex.psm == GS_PSM_T4);
        ts.fst = st.prim.fst;
        ts.linear = st.linearFilter;
        ts.on = true;
        return nullptr;
    }

    // VRAM pages the texels of a texture can come from (exact page arithmetic of GSMem::SwizzledSurface).
    static void TexturePages(const TexKey &k, std::vector<uint16_t> &out)
    {
        uint32_t pw = 64u, ph = 32u;
        if (k.psm == GS_PSM_T8)
        {
            pw = 128u;
            ph = 64u;
        }
        else if (k.psm == GS_PSM_T4)
        {
            pw = 128u;
            ph = 128u;
        }
        const uint32_t ppr = (k.tbw * 64u) / pw; // may be 0 (the oracle aliases the page rows then)
        const uint32_t base = k.tbp / 32u;
        const bool spill = (k.tbp % 32u) != 0u;
        const uint32_t rows = (k.h + ph - 1u) / ph, cols = (k.w + pw - 1u) / pw;
        std::bitset<kPages> set;
        for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t c = 0; c < cols; ++c)
            {
                const uint32_t p = (base + r * ppr + c) % kPages;
                set.set(p);
                if (spill)
                    set.set((p + 1u) % kPages);
            }
        out.clear();
        for (uint32_t p = 0; p < kPages; ++p)
            if (set.test(p))
                out.push_back(uint16_t(p));
    }

    id<MTLTexture> DecodeTexture(const TexKey &k, size_t &bytesOut)
    {
        using namespace GSMem;
        const uint32_t W = k.w, H = k.h;
        const bool idx = (k.psm == GS_PSM_T8 || k.psm == GS_PSM_T4);
        const uint8_t *src = nullptr;
        if (k.psm == GS_PSM_T8)
        {
            scratch8.resize(size_t(W) * H);
            const SwizzledSurface<P8> sw(k.tbp, k.tbw);
            for (uint32_t y = 0; y < H; ++y)
                for (uint32_t x = 0; x < W; ++x)
                    scratch8[size_t(y) * W + x] = static_cast<uint8_t>(sw.Read(vram, x, y));
            src = scratch8.data();
        }
        else if (k.psm == GS_PSM_T4)
        {
            scratch8.resize(size_t(W) * H);
            const SwizzledSurface<P4> sw(k.tbp, k.tbw);
            for (uint32_t y = 0; y < H; ++y)
                for (uint32_t x = 0; x < W; ++x)
                    scratch8[size_t(y) * W + x] = static_cast<uint8_t>(sw.Read(vram, x, y));
            src = scratch8.data();
        }
        else if (k.psm == GS_PSM_CT32)
        {
            scratch32.resize(size_t(W) * H);
            const SwizzledSurface<C32> sw(k.tbp, k.tbw);
            for (uint32_t y = 0; y < H; ++y)
                for (uint32_t x = 0; x < W; ++x)
                    scratch32[size_t(y) * W + x] = sw.Read(vram, x, y);
            src = reinterpret_cast<const uint8_t *>(scratch32.data());
        }
        else // CT24 + TEXA (applyTexa of the oracle)
        {
            scratch32.resize(size_t(W) * H);
            const SwizzledSurface<C24> sw(k.tbp, k.tbw);
            const uint32_t ta0 = k.texa & 0xFFu;
            const bool aem = (k.texa & 0x100u) != 0u;
            for (uint32_t y = 0; y < H; ++y)
                for (uint32_t x = 0; x < W; ++x)
                {
                    const uint32_t texel = sw.Read(vram, x, y);
                    const bool rgbZero = (texel & 0x00FFFFFFu) == 0u;
                    const uint32_t a = (aem && rgbZero) ? 0u : ta0;
                    scratch32[size_t(y) * W + x] = (texel & 0x00FFFFFFu) | (a << 24);
                }
            src = reinterpret_cast<const uint8_t *>(scratch32.data());
        }
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:(idx ? MTLPixelFormatR8Uint : MTLPixelFormatRGBA8Uint)
                                                                                     width:W
                                                                                    height:H
                                                                                 mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead;
        d.storageMode = MTLStorageModeShared;
        id<MTLTexture> tex = [device newTextureWithDescriptor:d];
        const size_t bpp = idx ? 1u : 4u;
        [tex replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:src bytesPerRow:W * bpp];
        bytesOut = size_t(W) * H * bpp;
        frame.texTexels += uint64_t(W) * H;
        return tex;
    }

    void EvictTextures()
    {
        constexpr size_t kBudget = 384u << 20;
        if (texBytes <= kBudget)
            return;
        std::vector<std::pair<uint64_t, TexKey>> order;
        order.reserve(texCache.size());
        for (const auto &[k, e] : texCache)
            order.emplace_back(e.lastUse, k);
        std::sort(order.begin(), order.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
        for (const auto &[use, k] : order)
        {
            if (texBytes <= kBudget / 2)
                break;
            auto it = texCache.find(k);
            texBytes -= it->second.bytes;
            texCache.erase(it);
        }
    }

    // Finds, validates (page epochs) or (re)decodes the draw's texture, closing the open run first
    // when the texture's pages overlap the run's target. Leaves the result in primTex / curPalOff.
    void PrepareTexture(const GSPrimitiveBatch &batch, const GSContext &ctx)
    {
        const uint64_t t0 = timing ? nowNs() : 0;
        auto [it, inserted] = texCache.try_emplace(ts.key);
        TexEntry &e = it->second;
        if (inserted)
            TexturePages(ts.key, e.pages);
        e.lastUse = ++useTick;
        if (run && !flushAll)
        {
            for (uint16_t p : e.pages)
                if (runPages.test(p))
                {
                    FlushRun(kFlushTexOverlap);
                    SelectRun(ctx);
                    break;
                }
        }
        else if (run && flushAll)
        {
            FlushRun(kFlushTexOverlap);
            SelectRun(ctx);
        }
        bool fresh = !e.tex;
        if (!fresh && e.checked != epoch)
        {
            for (uint16_t p : e.pages)
                if (pageEpoch[p] > e.epoch)
                {
                    fresh = true;
                    break;
                }
            if (!fresh)
                e.checked = epoch;
        }
        if (fresh)
        {
            size_t bytes = 0;
            id<MTLTexture> tex = DecodeTexture(ts.key, bytes);
            texBytes -= e.bytes;
            e.tex = tex;
            e.bytes = bytes;
            texBytes += bytes;
            e.epoch = epoch;
            e.checked = epoch;
            ++frame.texDecodes;
            primTex = tex;
            EvictTextures();
        }
        else
        {
            ++frame.texHits;
            primTex = e.tex;
        }

        curPalOff = 0;
        if (ts.indexed)
        {
            uint64_t gen = 0, key = 0;
            const uint32_t *pal = cpu->PaletteFor(batch.state, gen, key);
            const uint32_t count = (ts.key.psm == GS_PSM_T4) ? 16u : 256u;
            if (lastPalValid && gen == lastPalGen && key == lastPalKey)
            {
                curPalOff = lastPalOff;
                ++frame.paletteHits;
            }
            else
            {
                uint64_t h = 0xCBF29CE484222325ull ^ count;
                for (uint32_t i = 0; i < count; ++i)
                {
                    h ^= pal[i];
                    h *= 0x100000001B3ull;
                    h ^= h >> 29;
                }
                uint32_t slot = UINT32_MAX;
                auto pit = palMap.find(h);
                if (pit != palMap.end() && std::memcmp(static_cast<uint32_t *>(palBuf.contents) + size_t(pit->second) * 256u, pal, count * 4u) == 0)
                {
                    slot = pit->second;
                    ++frame.paletteHits;
                }
                else
                {
                    if (palUsed >= kPalSlots)
                    {
                        FlushRun(kFlushOther);
                        SelectRun(ctx);
                    }
                    slot = palUsed++;
                    std::memcpy(static_cast<uint32_t *>(palBuf.contents) + size_t(slot) * 256u, pal, count * 4u);
                    palMap[h] = slot;
                    ++frame.palettes;
                }
                curPalOff = slot * 256u;
                lastPalValid = true;
                lastPalGen = gen;
                lastPalKey = key;
                lastPalOff = curPalOff;
            }
        }
        if (timing)
            frame.texNs += nowNs() - t0;
    }

    // Start (or continue) the draw segment that binds `tex` for the primitives recorded next.
    void BindSegment(id<MTLTexture> tex)
    {
        if (!tex)
        {
            if (segments.empty())
            {
                segments.push_back({0u, dummyTex});
                curTex = dummyTex;
            }
            return;
        }
        if (segments.empty() || curTex != tex)
        {
            const uint32_t first = uint32_t(verts.size());
            if (!segments.empty() && segments.back().first == first)
                segments.back().tex = tex;
            else
                segments.push_back({first, tex});
            curTex = tex;
        }
    }

    // ---------------------------------------------------------------- textured sprites (G2b)
    struct SpriteGeom
    {
        bool ok = false;
        int un0x = 0, un0y = 0; // unclipped top-left (Raster::Sprite's unclippedX0/Y0)
        float w = 1.0f, h = 1.0f;
        float u0f = 0.0f, v0f = 0.0f, u1f = 0.0f, v1f = 0.0f;
        int gx0 = 0, gy0 = 0, gx1 = -1, gy1 = -1; // drawn (scissored) rectangle, inclusive
    };
    SpriteGeom sg;

    static float FabsQ(float q) { return (std::fabs(q) > 1.0e-8f) ? q : 1.0f; }

    // Raster::Sprite's setup, the oracle's own expressions (run under the draw's FPCR).
    void ComputeSpriteGeom(const GSPrimitiveBatch &batch)
    {
        using GSInternal::clampInt;
        const GSContext &ctx = batch.state.context;
        const GSVertex v0 = batch.vertices[0];
        const GSVertex v1 = batch.vertices[1];
        const int ofx = ctx.xyoffset.ofx >> 4, ofy = ctx.xyoffset.ofy >> 4;
        int x0 = static_cast<int>(v0.x) - ofx, y0 = static_cast<int>(v0.y) - ofy;
        int x1 = static_cast<int>(v1.x) - ofx, y1 = static_cast<int>(v1.y) - ofy;
        if (x0 > x1)
            std::swap(x0, x1);
        if (y0 > y1)
            std::swap(y0, y1);
        const int spanX = std::max(1, x1 - x0), spanY = std::max(1, y1 - y0);
        const int ux1 = x0 + spanX - 1, uy1 = y0 + spanY - 1;
        sg = SpriteGeom{};
        if (ux1 < ctx.scissor.x0 || x0 > ctx.scissor.x1 || uy1 < ctx.scissor.y0 || y0 > ctx.scissor.y1)
            return;
        sg.un0x = x0;
        sg.un0y = y0;
        sg.gx0 = clampInt(x0, ctx.scissor.x0, ctx.scissor.x1);
        sg.gy0 = clampInt(y0, ctx.scissor.y0, ctx.scissor.y1);
        sg.gx1 = clampInt(ux1, ctx.scissor.x0, ctx.scissor.x1);
        sg.gy1 = clampInt(uy1, ctx.scissor.y0, ctx.scissor.y1);
        sg.w = static_cast<float>(spanX);
        sg.h = static_cast<float>(spanY);
        if (ts.fst)
        {
            sg.u0f = static_cast<float>(v0.u >> 4);
            sg.v0f = static_cast<float>(v0.v >> 4);
            sg.u1f = static_cast<float>(v1.u >> 4);
            sg.v1f = static_cast<float>(v1.v >> 4);
        }
        else
        {
            const float q0 = FabsQ(v0.q);
            const float q1 = FabsQ(v1.q);
            sg.u0f = (v0.s / q0) * static_cast<float>(ts.texW);
            sg.v0f = (v0.t / q0) * static_cast<float>(ts.texH);
            sg.u1f = (v1.s / q1) * static_cast<float>(ts.texW);
            sg.v1f = (v1.t / q1) * static_cast<float>(ts.texH);
        }
        sg.ok = true;
    }

    // The texel-space coordinate of the sprite pixel `pix` on one axis (Raster::Sprite::columnAxis / row).
    static float SpriteCoord(int pix, int un0, float size, float a0, float a1, bool fst)
    {
        const float t = (static_cast<float>(pix - un0) + 0.5f) / size;
        const float tex = a0 + (a1 - a0) * t;
        if (fst)
        {
            const int fixed = static_cast<int>((tex * 16.0f) + 0.5f);
            return static_cast<float>(static_cast<uint16_t>(GSInternal::clampInt(fixed, 0, 0xFFFF))) / 16.0f;
        }
        return tex;
    }

    static int WrapIdx(int c, int size, uint32_t mode, uint32_t rmin, uint32_t rmax)
    {
        switch (mode & 3u)
        {
        case 0u: return int(uint32_t(c) & uint32_t(size - 1));
        case 1u: return c < 0 ? 0 : (c > size - 1 ? size - 1 : c);
        case 2u: return std::min(std::max(c, int(rmin)), int(rmax));
        default: return int((uint32_t(c) & rmin) | rmax);
        }
    }

    // The texels one axis of a sprite can read. The coordinate is monotone along the axis, so the
    // first and last drawn pixel bound every texel (REPEAT is only handled while no wrapping occurs,
    // REGION_REPEAT not at all; CLAMP and REGION_CLAMP are monotone). lo/hi are the wrapped extremes
    // including the bilinear neighbour, hi0 those of the nearest texel; fracHi is the bilinear weight of
    // the neighbour of the largest-coordinate column.
    struct AxisRange
    {
        bool ok = false;
        int lo = 0, hi = 0, hi0 = 0, wa = 0, wb = 0;
        bool ident = false;
        float fracHi = 0.0f;
        int hiEff = 0; // hi without a neighbour read whose bilinear weight is exactly zero
    };
    static AxisRange SpriteAxis(float cA, float cB, bool linear, int size, uint32_t mode, uint32_t rmin, uint32_t rmax)
    {
        AxisRange r;
        const auto c0 = [&](float c) { return linear ? static_cast<int>(std::floor(c - 0.5f)) : static_cast<int>(c); };
        const int a = c0(cA), b = c0(cB);
        if ((mode & 3u) == 3u)
            return r;
        if ((mode & 3u) == 0u && (std::min(a, b) < 0 || std::max(a, b) + (linear ? 1 : 0) > size - 1))
            return r;
        r.wa = WrapIdx(a, size, mode, rmin, rmax);
        r.wb = WrapIdx(b, size, mode, rmin, rmax);
        r.lo = std::min(r.wa, r.wb);
        r.hi0 = r.hi = std::max(r.wa, r.wb);
        if (linear)
        {
            const int na = WrapIdx(a + 1, size, mode, rmin, rmax), nb = WrapIdx(b + 1, size, mode, rmin, rmax);
            r.lo = std::min(r.lo, std::min(na, nb));
            r.hi = std::max(r.hi, std::max(na, nb));
            r.fracHi = (std::max(cA, cB) - 0.5f) - static_cast<float>(std::max(a, b));
        }
        r.ident = (a == r.wa && b == r.wb);
        r.hiEff = (linear && r.hi > r.hi0 && r.fracHi == 0.0f) ? r.hi0 : r.hi; // weight exactly 0: fma(x, 0, f00) == f00 whatever the neighbour holds
        r.ok = true;
        return r;
    }

    // Exactness proof for sampling a target texture directly: every texel read lies inside the target;
    // the one exception is the +1 bilinear neighbour just past the target edge when its weight is
    // exactly zero (always the case for FST coordinates, which are multiples of 1/16), so its value
    // cannot matter. A feedback sprite (source == destination) must in
    // addition be a 1:1 point-sampled copy, so each pixel is read before it is written.
    static bool AxisFits(const AxisRange &r, bool linear, int tdim, int pixA, int pixB, bool self)
    {
        if (!r.ok || r.lo < 0 || r.hi0 >= tdim)
            return false;
        if (self && (linear || !r.ident || r.wa != pixA || r.wb != pixB))
            return false;
        if (r.hi < tdim)
            return true;
        return linear && r.fracHi == 0.0f;
    }

    bool ColorStale(const Target &t) const
    {
        if (!t.valid)
            return true;
        PageRange r;
        TargetPages(t, false, r);
        for (uint32_t i = 0; i < r.count; ++i)
            if (pageEpoch[r.pages[i]] > t.syncEpoch)
                return true;
        return false;
    }

    // Samples a framebuffer target's own MTLTexture (the full-screen CT24/CT32 "previous frame" sprite)
    // instead of decoding the shadow. Leaves the texture in primTex / fbSrc; false = use the normal path.
    bool TryFbTexture(const GSContext &ctx)
    {
        static const bool dbg = std::getenv("PS2X_GS_METAL_DBG") != nullptr;
        fbSrc = false;
        const auto reject = [&](const char *why)
        {
            if (dbg)
            {
                const uint64_t n = ++frame.fallbacks[std::string("dbg_fb_") + why];
                if (n <= 3)
                {
                    std::fprintf(stderr, "[gsmtl-dbg] fb reject %s: tbp=%u tbw=%u psm=%u tex=%ux%u wms=%u wmt=%u lin=%d fst=%d | run fbp=%u zbp=%u fbw=%u | sprite %d,%d..%d,%d u=%.3f..%.3f v=%.3f..%.3f | targets:",
                                 why, ts.key.tbp, ts.key.tbw, ts.key.psm, ts.texW, ts.texH, ts.wms, ts.wmt, ts.linear, ts.fst, run ? run->fbp : 0u, run ? run->zbp : 0u,
                                 run ? run->fbw : 0u, sg.gx0, sg.gy0, sg.gx1, sg.gy1, sg.u0f, sg.u1f, sg.v0f, sg.v1f);
                    for (auto &t : targets)
                        std::fprintf(stderr, " (fbp=%u fbw=%u h=%u valid=%d stale=%d)", t->fbp, t->fbw, t->height, t->valid, ColorStale(*t));
                    std::fprintf(stderr, "\n");
                    std::fflush(stderr);
                }
            }
            return false;
        };
        if (!sg.ok || (ts.key.psm != GS_PSM_CT32 && ts.key.psm != GS_PSM_CT24) || ts.wms == 3u || ts.wmt == 3u || !run)
            return reject("psm_wrap");
        Target *T = nullptr;
        bool self = false;
        if ((run->fbp << 5) == ts.key.tbp && run->fbw == ts.key.tbw)
        {
            T = run;
            self = true;
        }
        else
        {
            for (auto &t : targets)
                if (t.get() != run && t->valid && (t->fbp << 5) == ts.key.tbp && t->fbw == ts.key.tbw && !ColorStale(*t))
                {
                    T = t.get();
                    break;
                }
            if (!T)
                return reject("no_target");
            PageRange r;
            pagesOfRect(T->fbp << 5, T->fbw, GS_PSM_CT32, 0u, 0u, T->width - 1u, T->height - 1u, r);
            if (OverlapsRun(r))
                return reject("run_overlap");
        }
        const bool fst = ts.fst;
        const float cu0 = SpriteCoord(sg.gx0, sg.un0x, sg.w, sg.u0f, sg.u1f, fst), cu1 = SpriteCoord(sg.gx1, sg.un0x, sg.w, sg.u0f, sg.u1f, fst);
        const float cv0 = SpriteCoord(sg.gy0, sg.un0y, sg.h, sg.v0f, sg.v1f, fst), cv1 = SpriteCoord(sg.gy1, sg.un0y, sg.h, sg.v0f, sg.v1f, fst);
        const AxisRange ru = SpriteAxis(cu0, cu1, ts.linear, int(ts.texW), ts.wms, ts.minU, ts.maxU);
        const AxisRange rv = SpriteAxis(cv0, cv1, ts.linear, int(ts.texH), ts.wmt, ts.minV, ts.maxV);
        if (!AxisFits(ru, ts.linear, int(T->width), sg.gx0, sg.gx1, self) || !AxisFits(rv, ts.linear, int(T->height), sg.gy0, sg.gy1, self))
            return reject(self ? "proof_self" : "proof");
        if (self)
        {
            if (!prims.empty())
            {
                FlushRun(kFlushTexOverlap);
                SelectRun(ctx);
                T = run;
            }
            if (!snapTex || snapTex.width != T->width || snapTex.height != T->height)
            {
                MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint
                                                                                             width:T->width
                                                                                            height:T->height
                                                                                         mipmapped:NO];
                d.usage = MTLTextureUsageShaderRead;
                d.storageMode = MTLStorageModePrivate;
                snapTex = [device newTextureWithDescriptor:d];
            }
            snapTarget = T;
            primTex = snapTex;
            ++frame.metalSelfPrims;
        }
        else
        {
            primTex = T->color;
            ++frame.metalFbPrims;
        }
        fbSrc = true;
        fbCt24 = ts.key.psm == GS_PSM_CT24;
        curPalOff = 0;
        return true;
    }

    // Do the texture's pages overlap the run target (a possible feedback read of pixels this draw writes)?
    bool TextureOverlapsRun() const
    {
        // The texels the sprite can read lie between those of its first and last pixel (monotone axes).
        const bool fst = ts.fst;
        const AxisRange ru = SpriteAxis(SpriteCoord(sg.gx0, sg.un0x, sg.w, sg.u0f, sg.u1f, fst), SpriteCoord(sg.gx1, sg.un0x, sg.w, sg.u0f, sg.u1f, fst),
                                        ts.linear, int(ts.texW), ts.wms, ts.minU, ts.maxU);
        const AxisRange rv = SpriteAxis(SpriteCoord(sg.gy0, sg.un0y, sg.h, sg.v0f, sg.v1f, fst), SpriteCoord(sg.gy1, sg.un0y, sg.h, sg.v0f, sg.v1f, fst),
                                        ts.linear, int(ts.texH), ts.wmt, ts.minV, ts.maxV);
        if (ru.ok && rv.ok && ru.hiEff < 2048 && rv.hiEff < 2048)
        {
            PageRange r;
            pagesOfRect(ts.key.tbp, ts.key.tbw, uint8_t(ts.key.psm), uint32_t(ru.lo), uint32_t(rv.lo), uint32_t(ru.hiEff), uint32_t(rv.hiEff), r);
            if (r.count < kPages)
            {
                for (uint32_t i = 0; i < r.count; ++i)
                    if (runPages.test(r.pages[i]))
                    {
                        return true;
                    }
                return false;
            }
        }
        std::vector<uint16_t> pages;
        TexturePages(ts.key, pages);
        for (uint16_t p : pages)
            if (runPages.test(p))
                return true;
        return false;
    }

    void FillTex(GPUPrim &p, const GSDrawState &state) const
    {
        p.tflags = 1u | (ts.fst ? 2u : 0u) | (ts.linear ? 4u : 0u) | (ts.indexed ? 8u : 0u) | (ts.tfx << 4) | (ts.tcc << 6) | (ts.wms << 8) | (ts.wmt << 10) |
                   (fbSrc ? (4096u | (fbCt24 ? 8192u : 0u)) : 0u);
        p.texdim = ts.texW | (ts.texH << 16);
        p.regU = ts.minU | (ts.maxU << 16);
        p.regV = ts.minV | (ts.maxV << 16);
        p.palOff = curPalOff;
        p.texa = uint32_t(state.texa.ta0) | (state.texa.aem ? 0x100u : 0u);
        p.fbw = ts.key.tbw;
    }

    const char *Eligible(const GSPrimitiveBatch &batch)
    {
        const GSDrawState &st = batch.state;
        const GSContext &ctx = st.context;
        ts.on = false;
        const bool tri = st.prim.type == GS_PRIM_TRIANGLE || st.prim.type == GS_PRIM_TRISTRIP || st.prim.type == GS_PRIM_TRIFAN;
        const bool sprite = st.prim.type == GS_PRIM_SPRITE;
        if (!tri && !sprite)
            return (st.prim.type == GS_PRIM_POINT) ? "point" : "line";
        if (ctx.frame.psm != GS_PSM_CT32)
            return "frame_psm";
        if (ctx.zbuf.psm != GS_PSM_Z24)
            return "zbuf_psm";
        if (ctx.frame.fbw == 0u || ctx.frame.fbw > 32u)
            return "fbw";
        if (ctx.scissor.x1 >= ctx.frame.fbw * 64u || ctx.scissor.y1 >= 2048u || ctx.scissor.x0 > ctx.scissor.x1 || ctx.scissor.y0 > ctx.scissor.y1)
            return "scissor";
        // frame and Z page spans must not overlap and must fit in VRAM
        const uint32_t pagesHigh = (uint32_t(ctx.scissor.y1) + 32u) / 32u;
        const uint32_t span = ctx.frame.fbw * pagesHigh;
        const uint32_t f0 = ctx.frame.fbp, f1 = f0 + span, z0 = ctx.zbuf.zbp, z1 = z0 + span;
        if (f1 > kPages || z1 > kPages)
            return "vram_range";
        if (f0 < z1 && z0 < f1)
            return "fz_overlap";
        const uint32_t vc = sprite ? 2u : 3u;
        // Triangle Z is interpolated in float (it must be exact); a sprite's Z is the integer of its second vertex.
        for (uint32_t i = 0; i < (sprite ? 0u : vc); ++i)
        {
            const double z = batch.vertices[i].z;
            if (static_cast<double>(static_cast<float>(z)) != z)
                return "z_not_float";
        }
        if (st.prim.tme)
            return ParseTexture(batch);
        return nullptr;
    }

    void Record(const GSPrimitiveBatch &batch, bool rtz)
    {
        using GSInternal::clampInt;
        const GSDrawState &state = batch.state;
        const GSContext &ctx = state.context;
        GPUPrim p{};
        p.flags = (rtz ? 4u : 0u) | (state.prim.fge ? 8u : 0u) | (state.prim.abe ? 16u : 0u) | (state.pabe ? 32u : 0u) |
                  ((ctx.fba & 1ull) ? 64u : 0u) | (ctx.zbuf.zmask ? 128u : 0u);
        static const bool s_fault = std::getenv("PS2X_GS_METAL_FAULT") != nullptr;
        if (s_fault)
            p.flags |= 256u;
        p.test = static_cast<uint32_t>(ctx.test);
        p.alpha = static_cast<uint32_t>(ctx.alpha & 0xFFu) | (static_cast<uint32_t>((ctx.alpha >> 32) & 0xFFu) << 8);
        p.fbmsk = ctx.frame.fbmsk;
        p.fogcol = uint32_t(state.fogR) | (uint32_t(state.fogG) << 8) | (uint32_t(state.fogB) << 16);

        int x0, y0, x1, y1; // inclusive pixel rectangle to cover
        const int ofx = ctx.xyoffset.ofx >> 4;
        const int ofy = ctx.xyoffset.ofy >> 4;
        if (state.prim.type == GS_PRIM_SPRITE)
        {
            // Raster::Sprite, untextured
            const GSVertex v0 = batch.vertices[0];
            const GSVertex v1 = batch.vertices[1];
            int sx0 = static_cast<int>(v0.x) - ofx;
            int sy0 = static_cast<int>(v0.y) - ofy;
            int sx1 = static_cast<int>(v1.x) - ofx;
            int sy1 = static_cast<int>(v1.y) - ofy;
            const uint32_t z1 = static_cast<uint32_t>(v1.z);
            if (sx0 > sx1)
                std::swap(sx0, sx1);
            if (sy0 > sy1)
                std::swap(sy0, sy1);
            const int spanX = std::max(1, sx1 - sx0);
            const int spanY = std::max(1, sy1 - sy0);
            const int ux1 = sx0 + spanX - 1;
            const int uy1 = sy0 + spanY - 1;
            if (ux1 < ctx.scissor.x0 || sx0 > ctx.scissor.x1 || uy1 < ctx.scissor.y0 || sy0 > ctx.scissor.y1)
                return;
            x0 = clampInt(sx0, ctx.scissor.x0, ctx.scissor.x1);
            y0 = clampInt(sy0, ctx.scissor.y0, ctx.scissor.y1);
            x1 = clampInt(ux1, ctx.scissor.x0, ctx.scissor.x1);
            y1 = clampInt(uy1, ctx.scissor.y0, ctx.scissor.y1);
            p.flags |= 1u;
            p.rgba0 = uint32_t(v1.r) | (uint32_t(v1.g) << 8) | (uint32_t(v1.b) << 16) | (uint32_t(v1.a) << 24);
            p.fog = v1.fog;
            p.spriteZ = z1;
            if (ts.on)
            {
                p.s0 = sg.u0f;
                p.t0 = sg.v0f;
                p.s1 = sg.u1f;
                p.t1 = sg.v1f;
                p.q0 = sg.w;
                p.q1 = sg.h;
                p.uv0 = uint32_t(sg.un0x);
                p.uv1 = uint32_t(sg.un0y);
                FillTex(p, state);
            }
        }
        else
        {
            // drawBatch triangle setup, verbatim
            const GSVertex &v0 = batch.vertices[0];
            const GSVertex &v1 = batch.vertices[1];
            const GSVertex &v2 = batch.vertices[2];
            float fx0 = v0.x - static_cast<float>(ofx);
            float fy0 = v0.y - static_cast<float>(ofy);
            float fx1 = v1.x - static_cast<float>(ofx);
            float fy1 = v1.y - static_cast<float>(ofy);
            float fx2 = v2.x - static_cast<float>(ofx);
            float fy2 = v2.y - static_cast<float>(ofy);
            int minX = static_cast<int>(std::floor(std::min({fx0, fx1, fx2})));
            int maxX = static_cast<int>(std::ceil(std::max({fx0, fx1, fx2})));
            int minY = static_cast<int>(std::floor(std::min({fy0, fy1, fy2})));
            int maxY = static_cast<int>(std::ceil(std::max({fy0, fy1, fy2})));
            minX = clampInt(minX, ctx.scissor.x0, ctx.scissor.x1);
            maxX = clampInt(maxX, ctx.scissor.x0, ctx.scissor.x1);
            minY = clampInt(minY, ctx.scissor.y0, ctx.scissor.y1);
            maxY = clampInt(maxY, ctx.scissor.y0, ctx.scissor.y1);
            float denom = (fy1 - fy2) * (fx0 - fx2) + (fx2 - fx1) * (fy0 - fy2);
            if (std::fabs(denom) < 0.001f)
                return;
            const float winding = (denom < 0.0f) ? -1.0f : 1.0f;
            const float invAbsDenom = 1.0f / std::fabs(denom);
            p.fx2 = fx2;
            p.fy2 = fy2;
            p.a0 = fy1 - fy2;
            p.b0 = fx2 - fx1;
            p.a1 = fy2 - fy0;
            p.b1 = fx0 - fx2;
            p.winding = winding;
            p.invAbsDenom = invAbsDenom;
            p.z0 = static_cast<float>(v0.z);
            p.z1 = static_cast<float>(v1.z);
            p.z2 = static_cast<float>(v2.z);
            p.rgba0 = uint32_t(v0.r) | (uint32_t(v0.g) << 8) | (uint32_t(v0.b) << 16) | (uint32_t(v0.a) << 24);
            p.rgba1 = uint32_t(v1.r) | (uint32_t(v1.g) << 8) | (uint32_t(v1.b) << 16) | (uint32_t(v1.a) << 24);
            p.rgba2 = uint32_t(v2.r) | (uint32_t(v2.g) << 8) | (uint32_t(v2.b) << 16) | (uint32_t(v2.a) << 24);
            p.fog = uint32_t(v0.fog) | (uint32_t(v1.fog) << 8) | (uint32_t(v2.fog) << 16);
            if (state.prim.iip)
                p.flags |= 2u;
            if (ts.on)
            {
                p.s0 = v0.s;
                p.t0 = v0.t;
                p.q0 = v0.q;
                p.s1 = v1.s;
                p.t1 = v1.t;
                p.q1 = v1.q;
                p.s2 = v2.s;
                p.t2 = v2.t;
                p.q2 = v2.q;
                p.uv0 = uint32_t(v0.u) | (uint32_t(v0.v) << 16);
                p.uv1 = uint32_t(v1.u) | (uint32_t(v1.v) << 16);
                p.uv2 = uint32_t(v2.u) | (uint32_t(v2.v) << 16);
                FillTex(p, state);
            }
            x0 = minX;
            x1 = maxX;
            y0 = minY;
            y1 = maxY;
        }
        if (x1 < x0 || y1 < y0)
            return;
        const uint32_t index = static_cast<uint32_t>(prims.size());
        prims.push_back(p);
        const float fx0 = float(x0), fy0 = float(y0), fx1 = float(x1 + 1), fy1 = float(y1 + 1);
        verts.push_back({fx0, fy0, index});
        verts.push_back({fx1, fy0, index});
        verts.push_back({fx0, fy1, index});
        verts.push_back({fx1, fy0, index});
        verts.push_back({fx1, fy1, index});
        verts.push_back({fx0, fy1, index});
        if (bx1 < bx0)
        {
            bx0 = x0;
            by0 = y0;
            bx1 = x1;
            by1 = y1;
        }
        else
        {
            bx0 = std::min(bx0, x0);
            by0 = std::min(by0, y0);
            bx1 = std::max(bx1, x1);
            by1 = std::max(by1, y1);
        }
    }
};

GSMetalBackend::GSMetalBackend() : m(std::make_unique<Impl>()) {}

GSMetalBackend::~GSMetalBackend()
{
    if (m)
    {
        m->FlushRun();
        Stats t = m->total;
        t.Add(m->frame);
        GSMetalPrintStats("total", t);
    }
}

std::unique_ptr<GSMetalBackend> GSMetalBackend::Create()
{
    @autoreleasepool
    {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device)
        {
            std::fprintf(stderr, "[gsmtl] no Metal device\n");
            return nullptr;
        }
        if (![device supportsFamily:MTLGPUFamilyApple1])
        {
            std::fprintf(stderr, "[gsmtl] device lacks framebuffer fetch (Apple GPU family)\n");
            return nullptr;
        }
        MTLCompileOptions *opts = [MTLCompileOptions new];
        if (@available(macOS 15.0, *))
            opts.mathMode = MTLMathModeSafe;
        else
        {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError *err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:[NSString stringWithUTF8String:kShaderSource] options:opts error:&err];
        if (!lib)
        {
            std::fprintf(stderr, "[gsmtl] shader compile failed: %s\n", err.localizedDescription.UTF8String);
            return nullptr;
        }
        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = [lib newFunctionWithName:@"vs_main"];
        pd.fragmentFunction = [lib newFunctionWithName:@"fs_main"];
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatR32Uint;
        pd.colorAttachments[1].pixelFormat = MTLPixelFormatR32Uint;
        id<MTLRenderPipelineState> pso = [device newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!pso)
        {
            std::fprintf(stderr, "[gsmtl] pipeline failed: %s\n", err.localizedDescription.UTF8String);
            return nullptr;
        }
        std::unique_ptr<GSMetalBackend> b(new GSMetalBackend());
        b->m->device = device;
        b->m->queue = [device newCommandQueue];
        b->m->pipeline = pso;
        b->m->selftestF = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"selftest_f"] error:&err];
        b->m->selftestZ = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"selftest_z"] error:&err];
        b->m->selftestT = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"selftest_t"] error:&err];
        b->m->cpu = std::make_unique<GSCpuBackend>();
        b->m->timing = std::getenv("PS2X_GS_METAL_TIMING") != nullptr;
        b->m->flushAll = std::getenv("PS2X_GS_METAL_FLUSH_ALL") != nullptr;
        b->m->palBuf = [device newBufferWithLength:size_t(Impl::kPalSlots) * 256u * 4u options:MTLResourceStorageModeShared];
        {
            MTLTextureDescriptor *dd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Uint width:1 height:1 mipmapped:NO];
            dd.usage = MTLTextureUsageShaderRead;
            dd.storageMode = MTLStorageModeShared;
            b->m->dummyTex = [device newTextureWithDescriptor:dd];
            const uint32_t zero = 0;
            [b->m->dummyTex replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&zero bytesPerRow:4];
        }
        std::fprintf(stderr, "[gsmtl] Metal backend on %s\n", device.name.UTF8String);
        std::fflush(stderr);
        return b;
    }
}

void GSMetalBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    m->FlushRun();
    m->vram = vram;
    m->vramSize = vramSize;
    m->cpu->Initialize(vram, vramSize);
    m->MarkAll();
}

void GSMetalBackend::Reset()
{
    m->FlushRun();
    m->cpu->Reset();
}

void GSMetalBackend::Submit(const GSPrimitiveBatch &batch)
{
    if (!m->vram || batch.vertexCount == 0u)
        return;
    ++m->frame.submits;
    const uint64_t t0 = m->timing ? nowNs() : 0;
    const auto fallback = [&](const char *reason)
    {
        m->FlushRun(kFlushFallback);
        m->cpu->Submit(batch);
        m->MarkCpuDraw(batch);
        ++m->frame.fallbackPrims;
        ++m->frame.fallbacks[reason];
        if (m->timing)
            m->frame.fallbackNs += nowNs() - t0;
    };
    m->fbSrc = false;
    if (const char *reason = m->Eligible(batch))
    {
        fallback(reason);
        return;
    }
    const GSContext &ctx = batch.state.context;
    const bool rtz = fpcrIsTowardZero(readFpcr());
    m->SelectRun(ctx);
    uint64_t tTex = 0;
    if (m->ts.on)
    {
        const uint64_t a = m->timing ? nowNs() : 0;
        bool prepared = false;
        if (batch.state.prim.type == GS_PRIM_SPRITE)
        {
            m->ComputeSpriteGeom(batch);
            prepared = m->TryFbTexture(ctx);
            if (!prepared && m->sg.ok && m->TextureOverlapsRun())
            {
                // The sprite may read pixels it has already written: only the oracle's live VRAM is exact.
                static const bool dbg = std::getenv("PS2X_GS_METAL_DBG") != nullptr;
                static int shown = 0;
                if (dbg && shown++ < 12)
                {
                    const auto &sgm = m->sg;
                    std::fprintf(stderr, "[gsmtl-dbg] tex_self: tbp=%u tbw=%u psm=%u tex=%ux%u wms=%u wmt=%u lin=%d fst=%d | run fbp=%u zbp=%u fbw=%u h=%u | sprite %d,%d..%d,%d u=%.3f..%.3f v=%.3f..%.3f\n",
                                 m->ts.key.tbp, m->ts.key.tbw, m->ts.key.psm, m->ts.texW, m->ts.texH, m->ts.wms, m->ts.wmt, m->ts.linear, m->ts.fst,
                                 m->run ? m->run->fbp : 0u, m->run ? m->run->zbp : 0u, m->run ? m->run->fbw : 0u, m->run ? m->run->height : 0u, sgm.gx0, sgm.gy0, sgm.gx1,
                                 sgm.gy1, sgm.u0f, sgm.u1f, sgm.v0f, sgm.v1f);
                    {
                        const bool fst2 = m->ts.fst;
                        const auto ru = Impl::SpriteAxis(Impl::SpriteCoord(sgm.gx0, sgm.un0x, sgm.w, sgm.u0f, sgm.u1f, fst2), Impl::SpriteCoord(sgm.gx1, sgm.un0x, sgm.w, sgm.u0f, sgm.u1f, fst2),
                                                         m->ts.linear, int(m->ts.texW), m->ts.wms, m->ts.minU, m->ts.maxU);
                        std::fprintf(stderr, "[gsmtl-dbg]   u ok=%d lo=%d hi=%d minU=%u maxU=%u runPages=%zu", ru.ok, ru.lo, ru.hi, m->ts.minU, m->ts.maxU, m->runPages.count());
                        PageRange pr;
                        pagesOfRect(m->ts.key.tbp, m->ts.key.tbw, uint8_t(m->ts.key.psm), uint32_t(ru.lo), 0u, uint32_t(ru.hi), 255u, pr);
                        for (uint32_t i = 0; i < pr.count; ++i)
                            if (m->runPages.test(pr.pages[i]))
                            {
                                std::fprintf(stderr, " overlap page %u", pr.pages[i]);
                                break;
                            }
                        std::fprintf(stderr, "\n");
                    }
                    std::fflush(stderr);
                }
                fallback("tex_self");
                return;
            }
        }
        if (!prepared)
            m->PrepareTexture(batch, ctx);
        tTex = m->timing ? nowNs() - a : 0;
    }
    ++m->frame.metalPrims;
    if (m->ts.on)
        ++m->frame.metalTexPrims;
    ++m->runPrims;
    m->BindSegment(m->ts.on ? m->primTex : nil);
    m->Record(batch, rtz);
    if (m->prims.size() >= 65536u)
        m->FlushRun(kFlushOther);
    if (m->timing)
        m->frame.recordNs += nowNs() - t0 - tTex;
}

void GSMetalBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    if (m->run)
    {
        // The CLUT is read from VRAM here (CSM1: a 16x16 block at CBP; CSM2: a row at COU/COV).
        PageRange r;
        if (tex0.csm == 0u)
            pagesOfRect(tex0.cbp, 1u, tex0.cpsm, 0u, 0u, 15u, 15u, r);
        else
            pagesOfRect(tex0.cbp, std::max<uint32_t>(texclut.cbw, 1u), tex0.cpsm, uint32_t(texclut.cou) << 4, texclut.cov,
                        (uint32_t(texclut.cou) << 4) + 255u, texclut.cov, r);
        m->FlushIfOverlaps(r, kFlushClut);
    }
    m->cpu->LoadClut(tex0, texclut);
}

void GSMetalBackend::BeginTransfer(const GSTransferCommand &command)
{
    const GSBitBltBuf &b = command.bitbltbuf;
    const bool rect = command.trxreg.rrw && command.trxreg.rrh;
    PageRange src, dst;
    src.count = dst.count = 0;
    // pages the transfer reads (local->host, local->local) and writes (host->local, local->local)
    if (command.direction == 1u || command.direction == 2u)
        pagesOfRect(b.sbp, b.sbw, b.spsm, command.trxpos.ssax, command.trxpos.ssay, command.trxpos.ssax + std::max<uint32_t>(command.trxreg.rrw, 1u) - 1u,
                    command.trxpos.ssay + std::max<uint32_t>(command.trxreg.rrh, 1u) - 1u, src);
    if ((command.direction == 0u || command.direction == 2u) && rect)
        pagesOfRect(b.dbp, b.dbw, b.dpsm, command.trxpos.dsax, command.trxpos.dsay, command.trxpos.dsax + command.trxreg.rrw - 1u,
                    command.trxpos.dsay + command.trxreg.rrh - 1u, dst);
    if (m->run)
    {
        m->FlushIfOverlaps(src, kFlushTransfer);
        m->FlushIfOverlaps(dst, kFlushTransfer);
        if (command.direction == 3u || command.direction > 2u)
            m->FlushRun(kFlushTransfer);
    }
    m->cpu->BeginTransfer(command);
    m->transfer = command;
    m->transferPages = dst;
    if (dst.count)
    {
        if (dst.count >= 512u)
            m->MarkAll();
        else
            m->MarkPages(dst);
    }
}

void GSMetalBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    if (m->run)
        m->FlushIfOverlaps(m->transferPages, kFlushTransfer);
    m->cpu->UploadImage(data, sizeBytes);
    if (m->transferPages.count >= 512u)
        m->MarkAll();
    else if (m->transferPages.count)
        m->MarkPages(m->transferPages);
}

void GSMetalBackend::Flush()
{
    m->FlushRun();
}

void GSMetalBackend::TextureFlush()
{
    m->FlushRun();
    m->cpu->TextureFlush();
}

void GSMetalBackend::Sync(GSSyncReason reason)
{
    m->FlushRun();
    m->cpu->Sync(reason);
}

PresentationFrame GSMetalBackend::Present(const GSPresentationRequest &request)
{
    m->FlushRun(kFlushPresent);
    ++m->frame.presents;
    static const uint32_t s_statsEvery = std::getenv("PS2X_GS_METAL_STATS") ? uint32_t(std::max(1, std::atoi(std::getenv("PS2X_GS_METAL_STATS")))) : 0u;
    if (s_statsEvery && m->frame.presents % s_statsEvery == 0u)
    {
        Stats t = m->total;
        t.Add(m->frame);
        GSMetalPrintStats("progress", t);
    }
    return m->cpu->Present(request);
}

bool GSMetalBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    m->FlushRun();
    const bool r = m->cpu->ClearFramebuffer(context, rgba);
    m->MarkAll();
    return r;
}

uint32_t GSMetalBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    m->FlushRun(kFlushReadback);
    return m->cpu->ConsumeLocalToHostBytes(dst, maxBytes);
}

uint32_t GSMetalBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    m->FlushRun(kFlushReadback);
    return m->cpu->ReadVram(psm, base, bw, x, y);
}

void GSMetalBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    m->FlushRun();
    m->cpu->WriteVram(psm, base, bw, x, y, value);
    m->MarkAll();
}

void GSMetalBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    m->FlushRun(kFlushReadback);
    m->cpu->SnapshotVram(out);
}

GSTransferSnapshot GSMetalBackend::GetTransferSnapshot() const
{
    m->FlushRun(kFlushReadback);
    return m->cpu->GetTransferSnapshot();
}

void GSMetalBackend::SetRunObserver(std::function<void(const RunInfo &)> observer)
{
    m->observer = std::move(observer);
}

void GSMetalBackend::MarkShadowChanged()
{
    m->FlushRun();
    m->MarkAll();
}

void GSMetalBackend::FlushRun()
{
    m->FlushRun();
}

GSMetalBackend::Stats GSMetalBackend::TakeFrameStats()
{
    Stats s = m->frame;
    m->total.Add(s);
    m->frame = Stats{};
    return s;
}

const GSMetalBackend::Stats &GSMetalBackend::TotalStats() const
{
    return m->total;
}

namespace
{
    // Host reference, compiled with the same -ffp-contract=on expressions as gs_cpu_backend.cpp.
    __attribute__((noinline)) void hostRefF(const float *in, uint32_t *out)
    {
        const float a = in[0], b = in[1], c = in[2];
        const float s = a + b;
        const float p = a * b;
        const float f = std::fma(a, b, c);
        std::memcpy(&out[0], &s, 4);
        std::memcpy(&out[1], &p, 4);
        std::memcpy(&out[2], &f, 4);
        const uint32_t c0 = uint32_t(in[3]) & 255u, c1 = (uint32_t(in[3]) >> 8) & 255u, c2 = 7u;
        const float w0 = a, w1 = b, w2 = c;
        out[3] = GSInternal::clampU8(static_cast<int>(c0 * w0 + c1 * w1 + c2 * w2));
    }

    __attribute__((noinline)) uint32_t hostRefZ(const float *z, const float *w)
    {
        const double z0 = z[0], z1 = z[1], z2 = z[2];
        const float w0 = w[0], w1 = w[1], w2 = w[2];
        double zz = z0 * w0 + z1 * w1 + z2 * w2;
        return static_cast<uint32_t>(zz + 0.5);
    }
}

uint64_t GSMetalBackend::SelfTest(uint32_t cases)
{
    uint64_t mismatches = 0;
    @autoreleasepool
    {
        std::mt19937 rng(12345u);
        std::uniform_real_distribution<float> coord(-64.0f, 1100.0f);
        std::uniform_real_distribution<float> weight(-0.0002f, 1.0002f);
        std::uniform_int_distribution<uint32_t> bits(0u, 0xFFFFFFFFu);
        std::uniform_int_distribution<uint32_t> zdist(0u, 0xFFFFFFu);
        std::vector<float> fin(size_t(cases) * 4u), zin(size_t(cases) * 4u), win(size_t(cases) * 4u);
        for (uint32_t i = 0; i < cases; ++i)
        {
            float *v = &fin[size_t(i) * 4u];
            switch (i % 4u)
            {
            case 0: // edge-function-like operands (1/16 fractions)
                v[0] = std::round(coord(rng) * 16.0f) / 16.0f + 0.5f;
                v[1] = -std::round(coord(rng) * 16.0f) / 16.0f;
                v[2] = coord(rng) * coord(rng);
                break;
            case 1: // weights
                v[0] = weight(rng);
                v[1] = weight(rng);
                v[2] = weight(rng);
                break;
            default: // random finite normal floats
                for (int k = 0; k < 3; ++k)
                {
                    uint32_t u;
                    do
                    {
                        u = bits(rng);
                        u = (u & 0x807FFFFFu) | ((((u >> 23) & 0xFFu) % 120u + 68u) << 23);
                    } while (false);
                    std::memcpy(&v[k], &u, 4);
                }
                break;
            }
            v[3] = float(bits(rng) & 0xFFFFu);
            float *z = &zin[size_t(i) * 4u];
            float *w = &win[size_t(i) * 4u];
            for (int k = 0; k < 3; ++k)
            {
                z[k] = float(zdist(rng));
                w[k] = weight(rng);
            }
            z[3] = w[3] = 0.0f;
        }
        for (uint32_t mode = 0; mode < 2; ++mode)
        {
            const uint32_t rtz = mode;
            id<MTLBuffer> bin = [m->device newBufferWithBytes:fin.data() length:fin.size() * 4u options:MTLResourceStorageModeShared];
            id<MTLBuffer> bout = [m->device newBufferWithLength:size_t(cases) * 16u options:MTLResourceStorageModeShared];
            id<MTLBuffer> bz = [m->device newBufferWithBytes:zin.data() length:zin.size() * 4u options:MTLResourceStorageModeShared];
            id<MTLBuffer> bw = [m->device newBufferWithBytes:win.data() length:win.size() * 4u options:MTLResourceStorageModeShared];
            id<MTLBuffer> bzo = [m->device newBufferWithLength:size_t(cases) * 4u options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cmd = [m->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:m->selftestF];
            [enc setBuffer:bin offset:0 atIndex:0];
            [enc setBuffer:bout offset:0 atIndex:1];
            [enc setBytes:&rtz length:4 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(cases, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc setComputePipelineState:m->selftestZ];
            [enc setBuffer:bz offset:0 atIndex:0];
            [enc setBuffer:bw offset:0 atIndex:1];
            [enc setBuffer:bzo offset:0 atIndex:2];
            [enc setBytes:&rtz length:4 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(cases, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            const int oldRound = std::fegetround();
            const uint64_t oldFpcr = readFpcr();
            if (rtz)
            {
                std::fesetround(FE_TOWARDZERO);
#if defined(__aarch64__)
                const uint64_t f = readFpcr() | (1ull << 24);
                __asm__ volatile("msr fpcr, %0" : : "r"(f));
#endif
            }
            uint64_t bad[5] = {0, 0, 0, 0, 0};
            const uint32_t *go = static_cast<const uint32_t *>(bout.contents);
            const uint32_t *gz = static_cast<const uint32_t *>(bzo.contents);
            for (uint32_t i = 0; i < cases; ++i)
            {
                uint32_t ref[4];
                hostRefF(&fin[size_t(i) * 4u], ref);
                for (int k = 0; k < 4; ++k)
                    if (ref[k] != go[size_t(i) * 4u + k])
                    {
                        if (bad[k] < 3u)
                            std::fprintf(stderr, "[gsmtl] selftest mismatch op=%d a=%a b=%a c=%a host=%08x gpu=%08x\n", k, fin[size_t(i) * 4u],
                                         fin[size_t(i) * 4u + 1], fin[size_t(i) * 4u + 2], ref[k], go[size_t(i) * 4u + k]);
                        ++bad[k];
                    }
                if (hostRefZ(&zin[size_t(i) * 4u], &win[size_t(i) * 4u]) != gz[i])
                    ++bad[4];
            }
#if defined(__aarch64__)
            __asm__ volatile("msr fpcr, %0" : : "r"(oldFpcr));
#endif
            std::fesetround(oldRound);
            const uint64_t sum = bad[0] + bad[1] + bad[2] + bad[3] + bad[4];
            mismatches += sum;
            std::fprintf(stderr, "[gsmtl] selftest %s cases=%u add=%llu mul=%llu fma=%llu interp=%llu z=%llu\n", rtz ? "rtz" : "rne", cases,
                         (unsigned long long)bad[0], (unsigned long long)bad[1], (unsigned long long)bad[2], (unsigned long long)bad[3],
                         (unsigned long long)bad[4]);
        }
        std::fflush(stderr);
    }
    return mismatches;
}

// ---------------------------------------------------------------------------------------------
// Differential test: random textured triangles through GSMetalBackend and the oracle GSCpuBackend
// on identical VRAM; the frame and Z planes must stay byte-identical after every draw.
// ---------------------------------------------------------------------------------------------
uint64_t GSMetalBackend::SelfTestDraws(uint32_t draws, uint32_t seed)
{
    std::unique_ptr<GSMetalBackend> mb = GSMetalBackend::Create();
    if (!mb)
        return ~0ull;
    std::vector<uint8_t> vramA(4u << 20), vramB(4u << 20);
    std::mt19937 rng(seed);
    auto rnd = [&](uint32_t n) { return n ? uint32_t(rng() % n) : 0u; };
    auto rndf = [&](float lo, float hi) { return lo + (hi - lo) * (float(rng() & 0xFFFFFFu) / float(0x1000000)); };
    auto refill = [&]()
    {
        for (size_t i = 0; i < vramA.size(); i += 4)
        {
            uint32_t v = static_cast<uint32_t>(rng());
            if (rng() & 1u)
                v = (v & 0x00FFFFFFu) | 0x80000000u; // opaque-ish
            std::memcpy(&vramA[i], &v, 4);
        }
        vramB = vramA;
        mb->Initialize(vramA.data(), uint32_t(vramA.size()));
    };
    GSCpuBackend ref;
    ref.Initialize(vramB.data(), uint32_t(vramB.size()));
    refill();

    uint64_t bad = 0;
    uint64_t counts[2] = {0, 0};
    for (uint32_t d = 0; d < draws; ++d)
    {
        if (d % 64u == 0u)
        {
            refill();
            ref.Initialize(vramB.data(), uint32_t(vramB.size()));
        }
        const bool rtz = (d & 1u) == 0u;
        GSPrimitiveBatch batch;
        GSDrawState &st = batch.state;
        GSContext &c = st.context;
        c.frame.fbp = 0;
        c.frame.fbw = 8;
        c.frame.psm = GS_PSM_CT32;
        c.frame.fbmsk = (rnd(8) == 0) ? static_cast<uint32_t>(rng()) : 0u;
        c.zbuf.zbp = 128;
        c.zbuf.psm = GS_PSM_Z24;
        c.zbuf.zmask = rnd(8) == 0;
        c.scissor = {0, 511, 0, 255};
        c.scissor.x0 = 0; c.scissor.x1 = 511; c.scissor.y0 = 0; c.scissor.y1 = 255;
        c.xyoffset.ofx = 28672;
        c.xyoffset.ofy = 30720;
        static const uint8_t psms[] = {GS_PSM_T8, GS_PSM_T4, GS_PSM_CT32, GS_PSM_CT24, GS_PSM_T8, GS_PSM_T4};
        // 0-3 triangle, 4-6 sprite (generic), 7-8 sprite sampling another framebuffer target, 9 feedback sprite (own target)
        const uint32_t kind = rnd(10);
        GSTex0Reg &t = c.tex0;
        t.psm = psms[rnd(6)];
        t.tw = uint8_t(4 + rnd(5));
        t.th = uint8_t(4 + rnd(5));
        t.tbw = uint8_t(1 + rnd(8));
        t.tbp0 = 8192u + rnd(7000u);
        if (rnd(4) == 0)
            t.tbp0 &= ~31u;
        t.tcc = rnd(4) != 0;
        t.tfx = rnd(5) == 0 ? rnd(4) : 0u;
        t.cbp = 4200u + rnd(2000u);
        t.cpsm = rnd(5) == 0 ? GS_PSM_CT16 : GS_PSM_CT32;
        t.csm = 0;
        t.csa = uint8_t(rnd(16));
        t.cld = 1;
        if (kind >= 7)
        {
            t.psm = rnd(2) ? GS_PSM_CT32 : GS_PSM_CT24;
            t.tbw = 8;
            t.tbp0 = (kind == 9) ? 0u : 2048u;
            t.tw = rnd(3) == 0 ? 10 : 9;
            t.th = rnd(3) == 0 ? 9 : 8;
        }
        st.textureWidth = uint16_t(1u << t.tw);
        st.textureHeight = uint16_t(1u << t.th);
        st.linearFilter = rnd(6) != 0;
        if (kind == 9)
            st.linearFilter = rnd(4) == 0;
        st.texa.ta0 = uint8_t(rng());
        st.texa.ta1 = uint8_t(rng());
        st.texa.aem = rnd(2) != 0;
        auto region = [&]() { return uint64_t(rnd(1024)); };
        uint64_t clamp = 0;
        uint64_t wms = rnd(7) == 0 ? rnd(4) : rnd(2), wmt = rnd(7) == 0 ? rnd(4) : rnd(2);
        clamp = wms | (wmt << 2);
        if (wms == 2u || wms == 3u)
            clamp |= (wms == 2u ? (region() & 0x3FFu) | (uint64_t(rnd(1024)) << 10) : (uint64_t(rnd(1024)) & 0x3C0u)) << 4;
        c.clamp = clamp | (region() << 24) | (region() << 34);
        c.clamp = (wms) | (wmt << 2) | ((wms == 2u ? uint64_t(rnd(8)) : uint64_t(rnd(1024) & 0x3F0u)) << 4) |
                  ((wms == 2u ? uint64_t(32 + rnd(224)) : uint64_t(rnd(64))) << 14) |
                  ((wmt == 2u ? uint64_t(rnd(8)) : uint64_t(rnd(1024) & 0x3F0u)) << 24) |
                  ((wmt == 2u ? uint64_t(32 + rnd(224)) : uint64_t(rnd(64))) << 34);
        if (kind >= 7 && rnd(6) != 0)
            c.clamp = (wms & 1u) | ((wmt & 1u) << 2);
        const uint32_t atst = rnd(8), afail = rnd(4);
        c.test = (rnd(3) != 0 ? 1ull : 0ull) | (uint64_t(atst) << 1) | (uint64_t(rnd(256)) << 4) | (uint64_t(afail) << 12) |
                 (uint64_t(rnd(4) == 0) << 14) | (uint64_t(rnd(2)) << 15) | (uint64_t(1) << 16) | (uint64_t(1 + rnd(3)) << 17);
        c.alpha = uint64_t(rnd(3)) | (uint64_t(rnd(3)) << 2) | (uint64_t(rnd(3)) << 4) | (uint64_t(rnd(3)) << 6) | (uint64_t(rnd(256)) << 32);
        st.pabe = rnd(8) == 0;
        st.fogR = uint8_t(rng());
        st.fogG = uint8_t(rng());
        st.fogB = uint8_t(rng());
        st.prim.type = kind >= 4 ? GS_PRIM_SPRITE : (rnd(2) ? GS_PRIM_TRIANGLE : GS_PRIM_TRISTRIP);
        st.prim.iip = rnd(4) != 0;
        st.prim.tme = true;
        st.prim.fge = rnd(3) != 0;
        st.prim.abe = rnd(3) != 0;
        st.prim.fst = rnd(6) == 0 || (kind >= 7 && rnd(2) == 0);
        batch.vertexCount = kind >= 4 ? 2 : 3;
        const float cx = rndf(1792.0f + 20.0f, 1792.0f + 490.0f), cy = rndf(1920.0f + 10.0f, 1920.0f + 240.0f);
        const float ext = rnd(5) == 0 ? 160.0f : 40.0f;
        for (int i = 0; i < 3; ++i)
        {
            GSVertex &v = batch.vertices[i];
            v.x = std::round((cx + rndf(-ext, ext)) * 16.0f) / 16.0f;
            v.y = std::round((cy + rndf(-ext, ext)) * 16.0f) / 16.0f;
            v.z = double(rnd(0x1000000u));
            v.r = uint8_t(rng());
            v.g = uint8_t(rng());
            v.b = uint8_t(rng());
            v.a = uint8_t(rng());
            v.fog = uint8_t(rng());
            float q = rndf(0.25f, 6.0f);
            if (rnd(40) == 0)
                q = rndf(-3.0f, 3.0f);
            if (rnd(200) == 0)
                q = rndf(-1e-7f, 1e-7f);
            v.q = q;
            v.s = rndf(-0.5f, 2.5f) * q;
            v.t = rndf(-0.5f, 2.5f) * q;
            v.u = uint16_t(rnd(1024u * 16u));
            v.v = uint16_t(rnd(1024u * 16u));
        }
        if (kind >= 4)
        {
            GSVertex &a = batch.vertices[0];
            GSVertex &b = batch.vertices[1];
            int rx0, ry0, rw, rh;
            if (kind >= 7)
            {
                if (rnd(4) == 0)
                {
                    rx0 = 0; ry0 = 0; rw = 512; rh = 256;
                }
                else
                {
                    rx0 = int(rnd(300)); ry0 = int(rnd(150)); rw = 1 + int(rnd(212)); rh = 1 + int(rnd(106));
                }
            }
            else
            {
                rx0 = int(rnd(500)); ry0 = int(rnd(250));
                rw = 1 + int(rnd(rnd(3) == 0 ? 400 : 60));
                rh = 1 + int(rnd(rnd(3) == 0 ? 200 : 40));
            }
            a.x = 1792.0f + float(rx0); a.y = 1920.0f + float(ry0);
            b.x = 1792.0f + float(rx0 + rw); b.y = 1920.0f + float(ry0 + rh);
            if (kind < 7 && rnd(8) == 0)
                std::swap(a.x, b.x);
            if (kind < 7 && rnd(8) == 0)
                std::swap(a.y, b.y);
            if (rnd(2) == 0)
            {
                a.z = double(rng()); // full 32-bit sprite Z (not float-exact, above 2^24)
                b.z = double(rng());
            }
            if (kind >= 7)
            {
                // 1:1 texel = pixel copy (the game's full-screen feedback sprite), optionally offset / scaled
                const uint32_t j = rnd(4);
                const int du = (j == 2) ? int(rnd(49)) - 24 : 0, dv = (j == 2) ? int(rnd(49)) - 24 : 0;
                const int su = (j == 3) ? int(rnd(40)) - 20 : 0, sv = (j == 3) ? int(rnd(40)) - 20 : 0;
                const int fu0 = rx0 * 16 + du, fv0 = ry0 * 16 + dv, fu1 = (rx0 + rw + su) * 16 + du, fv1 = (ry0 + rh + sv) * 16 + dv;
                a.u = uint16_t(std::max(fu0, 0)); a.v = uint16_t(std::max(fv0, 0));
                b.u = uint16_t(std::max(fu1, 0)); b.v = uint16_t(std::max(fv1, 0));
                const float tw = float(st.textureWidth), th = float(st.textureHeight);
                a.q = 1.0f; b.q = 1.0f;
                a.s = (float(fu0) / 16.0f) / tw; a.t = (float(fv0) / 16.0f) / th;
                b.s = (float(fu1) / 16.0f) / tw; b.t = (float(fv1) / 16.0f) / th;
            }
        }
        // CLUT: both backends load it from their (identical) VRAM
        GSTexClutReg tc{};
        mb->LoadClut(t, tc);
        ref.LoadClut(t, tc);
        if (kind == 7 || kind == 8)
        {
            // make the "other" framebuffer (fbp 64, the next buffer: TBP = FBP + 2048 blocks) a valid GPU-owned target
            for (int k = 0; k < 2; ++k)
            {
                GSPrimitiveBatch bb{};
                GSContext &bc = bb.state.context;
                bc.frame.fbp = 64; bc.frame.fbw = 8; bc.frame.psm = GS_PSM_CT32; bc.frame.fbmsk = 0;
                bc.zbuf.zbp = 192; bc.zbuf.psm = GS_PSM_Z24; bc.zbuf.zmask = false;
                bc.scissor.x0 = 0; bc.scissor.x1 = 511; bc.scissor.y0 = 0; bc.scissor.y1 = 255;
                bc.xyoffset = c.xyoffset;
                bc.test = (1ull << 16) | (1ull << 17);
                bb.state.prim.type = GS_PRIM_SPRITE;
                bb.vertexCount = 2;
                const int bx0 = int(rnd(400)), by0 = int(rnd(200));
                bb.vertices[0].x = 1792.0f + float(bx0); bb.vertices[0].y = 1920.0f + float(by0);
                bb.vertices[1].x = 1792.0f + float(bx0 + 1 + int(rnd(110))); bb.vertices[1].y = 1920.0f + float(by0 + 1 + int(rnd(56)));
                for (int i = 0; i < 2; ++i)
                {
                    bb.vertices[i].z = double(rnd(0x1000000u));
                    bb.vertices[i].r = uint8_t(rng()); bb.vertices[i].g = uint8_t(rng()); bb.vertices[i].b = uint8_t(rng()); bb.vertices[i].a = uint8_t(rng());
                }
                mb->Submit(bb);
                ref.Submit(bb);
            }
            mb->FlushRun();
        }
        const uint64_t oldFpcr = readFpcr();
        const int oldRound = std::fegetround();
        if (rtz)
        {
            std::fesetround(FE_TOWARDZERO);
#if defined(__aarch64__)
            const uint64_t f = readFpcr() | (1ull << 24);
            __asm__ volatile("msr fpcr, %0" : : "r"(f));
#endif
        }
        mb->Submit(batch);
        ref.Submit(batch);
#if defined(__aarch64__)
        __asm__ volatile("msr fpcr, %0" : : "r"(oldFpcr));
#endif
        std::fesetround(oldRound);
        mb->FlushRun();
        ++counts[rtz ? 1 : 0];
        if (std::memcmp(vramA.data(), vramB.data(), vramA.size()) != 0)
        {
            ++bad;
            size_t first = 0, n = 0;
            for (size_t i = 0; i < vramA.size(); ++i)
                if (vramA[i] != vramB[i])
                {
                    if (!n)
                        first = i;
                    ++n;
                }
            if (bad <= 12)
                std::fprintf(stderr, "[gsmtl] draw-test MISMATCH draw=%u kind=%u rtz=%d psm=%u tw=%u th=%u tbw=%u tfx=%u tcc=%u lin=%d fst=%d wms=%llu wmt=%llu prim=%d iip=%d bytes=%zu first=0x%zx\n",
                             d, kind, rtz, t.psm, t.tw, t.th, t.tbw, t.tfx, t.tcc, st.linearFilter, st.prim.fst, (unsigned long long)wms,
                             (unsigned long long)wmt, st.prim.type, st.prim.iip, n, first);
            vramA = vramB; // continue from the oracle's state
            mb->MarkShadowChanged();
        }
    }
    std::fprintf(stderr, "[gsmtl] draw-test draws=%u (rtz=%llu default=%llu) mismatching=%llu\n", draws, (unsigned long long)counts[1],
                 (unsigned long long)counts[0], (unsigned long long)bad);
    {
        mb->FlushRun();
        GSMetalBackend::Stats fs = mb->TakeFrameStats();
        GSMetalBackend::Stats ts2 = mb->TotalStats();
        ts2.Add(fs);
        GSMetalPrintStats("drawtest", ts2);
    }
    return bad;
}
