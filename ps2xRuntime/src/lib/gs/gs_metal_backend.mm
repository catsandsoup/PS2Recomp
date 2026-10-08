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
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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
    };
    static_assert(sizeof(GPUPrim) == 84, "GPUPrim layout");

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
                      uint fb [[color(0)]], uint zb [[color(1)]])
{
    const Prim P = prims[in.prim];
    const bool rtz = (P.flags & 4u) != 0u;
    uint r, g, b, a, fog, z;
    if ((P.flags & 1u) != 0u) {
        r = P.rgba0 & 0xFFu; g = (P.rgba0 >> 8) & 0xFFu; b = (P.rgba0 >> 16) & 0xFFu; a = P.rgba0 >> 24;
        fog = P.fog & 0xFFu;
        z = P.spriteZ;
    } else {
        const float px = in.pos.x, py = in.pos.y; // pixel centre = integer + 0.5
        const float dx = f_sub(px, P.fx2, rtz), dy = f_sub(py, P.fy2, rtz);
        const float w0 = f_mul(f_mul(f_fma(P.a0, dx, f_mul(P.b0, dy, rtz), rtz), P.winding, rtz), P.invAbsDenom, rtz);
        const float w1 = f_mul(f_mul(f_fma(P.a1, dx, f_mul(P.b1, dy, rtz), rtz), P.winding, rtz), P.invAbsDenom, rtz);
        const float w2 = f_sub(f_sub(1.0f, w0, rtz), w1, rtz);
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
    fallbackPrims += o.fallbackPrims;
    runs += o.runs;
    targetUploads += o.targetUploads;
    readbackPixels += o.readbackPixels;
    gpuWaitNs += o.gpuWaitNs;
    for (const auto &[k, v] : o.fallbacks)
        fallbacks[k] += v;
}

void GSMetalPrintStats(const char *tag, const GSMetalBackend::Stats &s)
{
    std::fprintf(stderr, "[gsmtl] %s submits=%llu metal=%llu fallback=%llu metal_frac=%.4f runs=%llu uploads=%llu readback_px=%llu gpu_wait_ms=%.1f\n",
                 tag, (unsigned long long)s.submits, (unsigned long long)s.metalPrims, (unsigned long long)s.fallbackPrims,
                 s.submits ? double(s.metalPrims) / double(s.submits) : 0.0, (unsigned long long)s.runs,
                 (unsigned long long)s.targetUploads, (unsigned long long)s.readbackPixels, double(s.gpuWaitNs) / 1e6);
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

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLRenderPipelineState> pipeline = nil;
    id<MTLComputePipelineState> selftestF = nil;
    id<MTLComputePipelineState> selftestZ = nil;

    std::unique_ptr<GSCpuBackend> cpu;
    uint8_t *vram = nullptr;
    uint32_t vramSize = 0;

    std::array<uint64_t, kPages> pageEpoch{};
    uint64_t epoch = 1;
    std::vector<std::unique_ptr<Target>> targets;

    // open run
    Target *run = nullptr;
    std::vector<GPUVertex> verts;
    std::vector<GPUPrim> prims;
    int bx0 = 0, by0 = 0, bx1 = -1, by1 = -1;
    uint32_t runPrims = 0;

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

    void FlushRun()
    {
        if (!run)
            return;
        Target &t = *run;
        @autoreleasepool
        {
            const bool stale = TargetStale(t);
            if (!prims.empty() || stale)
            {
                id<MTLCommandBuffer> cmd = [queue commandBuffer];
                const size_t plane = size_t(t.width) * t.height * 4u;
                if (stale)
                {
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
                }
                const bool draw = !prims.empty() && bx1 >= bx0 && by1 >= by0;
                if (draw)
                {
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
                    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:verts.size()];
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
        verts.clear();
        prims.clear();
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

    const char *Eligible(const GSPrimitiveBatch &batch) const
    {
        const GSDrawState &st = batch.state;
        const GSContext &ctx = st.context;
        const bool tri = st.prim.type == GS_PRIM_TRIANGLE || st.prim.type == GS_PRIM_TRISTRIP || st.prim.type == GS_PRIM_TRIFAN;
        const bool sprite = st.prim.type == GS_PRIM_SPRITE;
        if (!tri && !sprite)
            return (st.prim.type == GS_PRIM_POINT) ? "point" : "line";
        if (st.prim.tme)
            return sprite ? "textured_sprite" : "textured_tri";
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
        for (uint32_t i = 0; i < vc; ++i)
        {
            const double z = batch.vertices[i].z;
            if (static_cast<double>(static_cast<float>(z)) != z)
                return "z_not_float";
        }
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
        b->m->cpu = std::make_unique<GSCpuBackend>();
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
    if (const char *reason = m->Eligible(batch))
    {
        m->FlushRun();
        m->cpu->Submit(batch);
        m->MarkCpuDraw(batch);
        ++m->frame.fallbackPrims;
        ++m->frame.fallbacks[reason];
        return;
    }
    const GSContext &ctx = batch.state.context;
    const uint32_t needHeight = uint32_t(ctx.scissor.y1) + 1u;
    Impl::Target *t = nullptr;
    // reuse the open run's target when it matches and is tall enough
    if (m->run && m->run->fbp == ctx.frame.fbp && m->run->zbp == ctx.zbuf.zbp && m->run->fbw == ctx.frame.fbw && m->run->height >= needHeight)
        t = m->run;
    else
    {
        m->FlushRun();
        t = m->GetTarget(ctx.frame.fbp, ctx.zbuf.zbp, ctx.frame.fbw, needHeight);
        m->run = t;
    }
    ++m->frame.metalPrims;
    ++m->runPrims;
    m->Record(batch, fpcrIsTowardZero(readFpcr()));
    if (m->prims.size() >= 65536u)
        m->FlushRun();
}

void GSMetalBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    m->FlushRun();
    m->cpu->LoadClut(tex0, texclut);
}

void GSMetalBackend::BeginTransfer(const GSTransferCommand &command)
{
    m->FlushRun();
    m->cpu->BeginTransfer(command);
    m->transfer = command;
    m->transferPages.count = 0;
    const GSBitBltBuf &b = command.bitbltbuf;
    if ((command.direction == 0u || command.direction == 2u) && command.trxreg.rrw && command.trxreg.rrh)
    {
        pagesOfRect(b.dbp, b.dbw, b.dpsm, command.trxpos.dsax, command.trxpos.dsay,
                    command.trxpos.dsax + command.trxreg.rrw - 1u, command.trxpos.dsay + command.trxreg.rrh - 1u, m->transferPages);
        if (m->transferPages.count >= 512u)
            m->MarkAll();
        else
            m->MarkPages(m->transferPages);
    }
}

void GSMetalBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    m->FlushRun();
    m->cpu->UploadImage(data, sizeBytes);
    if (m->transferPages.count >= 512u)
        m->MarkAll();
    else
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
    m->FlushRun();
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
    m->FlushRun();
    return m->cpu->ConsumeLocalToHostBytes(dst, maxBytes);
}

uint32_t GSMetalBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    m->FlushRun();
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
    m->FlushRun();
    m->cpu->SnapshotVram(out);
}

GSTransferSnapshot GSMetalBackend::GetTransferSnapshot() const
{
    m->FlushRun();
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
