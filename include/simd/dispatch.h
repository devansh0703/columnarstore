#pragma once

#include <immintrin.h>
#include <cpuid.h>
#include <cstdint>
#include <atomic>

namespace columnar {
namespace simd {

enum class SimdLevel : uint8_t {
    Scalar = 0,
    SSE2 = 1,
    SSE42 = 2,
    AVX = 3,
    AVX2 = 4,
    AVX512F = 5,
    AVX512BW = 6,
    AVX512VL = 7
};

struct CpuFeatures {
    bool sse2 = false;
    bool sse3 = false;
    bool ssse3 = false;
    bool sse41 = false;
    bool sse42 = false;
    bool avx = false;
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512bw = false;
    bool avx512vl = false;
    bool avx512dq = false;
    bool avx512cd = false;
    bool bmi1 = false;
    bool bmi2 = false;
    bool lzcnt = false;
    bool popcnt = false;
};

inline CpuFeatures DetectCpuFeatures() {
    CpuFeatures features;
    unsigned int eax, ebx, ecx, edx;
    
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    features.sse2 = (edx >> 26) & 1;
    features.sse3 = ecx & 1;
    features.ssse3 = (ecx >> 9) & 1;
    features.sse41 = (ecx >> 19) & 1;
    features.sse42 = (ecx >> 20) & 1;
    features.avx = (ecx >> 28) & 1;
    features.popcnt = (ecx >> 23) & 1;
    
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        features.avx2 = (ebx >> 5) & 1;
        features.fma = (ebx >> 12) & 1;
        features.bmi1 = (ebx >> 3) & 1;
        features.bmi2 = (ebx >> 8) & 1;
        features.avx512f = (ebx >> 16) & 1;
        features.avx512bw = (ebx >> 30) & 1;
        features.avx512vl = (ebx >> 31) & 1;
        features.avx512dq = (ebx >> 17) & 1;
        features.avx512cd = (ebx >> 28) & 1;
        features.lzcnt = (ecx >> 5) & 1;
    }
    
    return features;
}

inline SimdLevel GetBestSimdLevel() {
    static std::atomic<SimdLevel> cached{SimdLevel::Scalar};
    SimdLevel level = cached.load(std::memory_order_relaxed);
    if (level != SimdLevel::Scalar) return level;
    
    CpuFeatures f = DetectCpuFeatures();
    if (f.avx512f && f.avx512bw && f.avx512vl) level = SimdLevel::AVX512VL;
    else if (f.avx512f && f.avx512bw) level = SimdLevel::AVX512BW;
    else if (f.avx512f) level = SimdLevel::AVX512F;
    else if (f.avx2) level = SimdLevel::AVX2;
    else if (f.avx) level = SimdLevel::AVX;
    else if (f.sse42) level = SimdLevel::SSE42;
    else if (f.sse2) level = SimdLevel::SSE2;
    
    cached.store(level, std::memory_order_relaxed);
    return level;
}

template<SimdLevel Level>
struct SimdOps;

#ifdef __AVX512F__
template<>
struct SimdOps<SimdLevel::AVX512VL> {
    static constexpr size_t Width = 64;
    using Vec = __m512i;
    using Mask = __mmask64;

    static Vec Load(const void* ptr) { return _mm512_loadu_si512(ptr); }
    static void Store(void* ptr, Vec v) { _mm512_storeu_si512(ptr, v); }
    static Vec Set1(int32_t v) { return _mm512_set1_epi32(v); }
    static Vec Set1(int64_t v) { return _mm512_set1_epi64(v); }
    static __m512 Set1(float v) { return _mm512_set1_ps(v); }
    static __m512d Set1(double v) { return _mm512_set1_pd(v); }

    static Mask CmpEq(Vec a, Vec b) { return _mm512_cmpeq_epi32_mask(a, b); }
    static __mmask8 CmpEq64(Vec a, Vec b) { return _mm512_cmpeq_epi64_mask(a, b); }
    static __mmask16 CmpEqPs(__m512 a, __m512 b) { return _mm512_cmpeq_ps_mask(a, b); }
    static __mmask8 CmpEqPd(__m512d a, __m512d b) { return _mm512_cmpeq_pd_mask(a, b); }

    static Mask CmpLt(Vec a, Vec b) { return _mm512_cmplt_epi32_mask(a, b); }
    static __mmask8 CmpLt64(Vec a, Vec b) { return _mm512_cmplt_epi64_mask(a, b); }
    static __mmask16 CmpLtPs(__m512 a, __m512 b) { return _mm512_cmplt_ps_mask(a, b); }
    static __mmask8 CmpLtPd(__m512d a, __m512d b) { return _mm512_cmplt_pd_mask(a, b); }

    static Mask CmpLe(Vec a, Vec b) { return _mm512_cmple_epi32_mask(a, b); }
    static __mmask8 CmpLe64(Vec a, Vec b) { return _mm512_cmple_epi64_mask(a, b); }
    static __mmask16 CmpLePs(__m512 a, __m512 b) { return _mm512_cmple_ps_mask(a, b); }
    static __mmask8 CmpLePd(__m512d a, __m512d b) { return _mm512_cmple_pd_mask(a, b); }

    static Vec Add(Vec a, Vec b) { return _mm512_add_epi32(a, b); }
    static Vec Add64(Vec a, Vec b) { return _mm512_add_epi64(a, b); }
    static __m512 AddPs(__m512 a, __m512 b) { return _mm512_add_ps(a, b); }
    static __m512d AddPd(__m512d a, __m512d b) { return _mm512_add_pd(a, b); }

    static Vec Sub(Vec a, Vec b) { return _mm512_sub_epi32(a, b); }
    static Vec Sub64(Vec a, Vec b) { return _mm512_sub_epi64(a, b); }
    static __m512 SubPs(__m512 a, __m512 b) { return _mm512_sub_ps(a, b); }
    static __m512d SubPd(__m512d a, __m512d b) { return _mm512_sub_pd(a, b); }

    static Vec Mul(Vec a, Vec b) { return _mm512_mullo_epi32(a, b); }
    static Vec Mul64(Vec a, Vec b) { return _mm512_mullo_epi64(a, b); }
    static __m512 MulPs(__m512 a, __m512 b) { return _mm512_mul_ps(a, b); }
    static __m512d MulPd(__m512d a, __m512d b) { return _mm512_mul_pd(a, b); }

    static Vec And(Vec a, Vec b) { return _mm512_and_si512(a, b); }
    static Vec Or(Vec a, Vec b) { return _mm512_or_si512(a, b); }
    static Vec Xor(Vec a, Vec b) { return _mm512_xor_si512(a, b); }
    static Vec Not(Vec a) { return _mm512_xor_si512(a, _mm512_set1_epi32(-1)); }

    static int PopCount(Mask m) { return _mm_popcnt_u64(m); }
    static Vec Compress(Vec a, Mask m) { return _mm512_maskz_compress_epi32(m, a); }
    static Vec Expand(Vec a, Mask m) { return _mm512_maskz_expand_epi32(m, a); }

    static Vec LoadMasked(const void* ptr, Mask m, Vec /*def*/) {
        return _mm512_maskz_loadu_epi32(m, ptr);
    }
    static void StoreMasked(void* ptr, Mask m, Vec v) {
        _mm512_mask_storeu_epi32(ptr, m, v);
    }
};

template<>
struct SimdOps<SimdLevel::AVX512BW> : SimdOps<SimdLevel::AVX512VL> {};

template<>
struct SimdOps<SimdLevel::AVX512F> : SimdOps<SimdLevel::AVX512VL> {};
#endif // __AVX512F__

template<>
struct SimdOps<SimdLevel::AVX2> {
    static constexpr size_t Width = 32;
    using Vec = __m256i;
    using Mask = int;
    
    static Vec Load(const void* ptr) { return _mm256_loadu_si256(static_cast<const __m256i*>(ptr)); }
    static void Store(void* ptr, Vec v) { _mm256_storeu_si256(static_cast<__m256i*>(ptr), v); }
    static Vec Set1(int32_t v) { return _mm256_set1_epi32(v); }
    static Vec Set1(int64_t v) { return _mm256_set1_epi64x(v); }
    static __m256 Set1(float v) { return _mm256_set1_ps(v); }
    static __m256d Set1(double v) { return _mm256_set1_pd(v); }
    
    static Mask CmpEq(Vec a, Vec b) {
        return _mm256_movemask_epi8(_mm256_cmpeq_epi32(a, b));
    }
    static Mask CmpLt(Vec a, Vec b) {
        return _mm256_movemask_epi8(_mm256_cmpgt_epi32(b, a));
    }
    static Mask CmpLe(Vec a, Vec b) {
        return _mm256_movemask_epi8(_mm256_cmpgt_epi32(_mm256_sub_epi32(b, a), _mm256_setzero_si256()));
    }
    static __m256 CmpEqPs(__m256 a, __m256 b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }
    static __m256d CmpEqPd(__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ); }
    
    static Vec Add(Vec a, Vec b) { return _mm256_add_epi32(a, b); }
    static Vec Add64(Vec a, Vec b) { return _mm256_add_epi64(a, b); }
    static __m256 AddPs(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
    static __m256d AddPd(__m256d a, __m256d b) { return _mm256_add_pd(a, b); }
    
    static Vec Sub(Vec a, Vec b) { return _mm256_sub_epi32(a, b); }
    static Vec Sub64(Vec a, Vec b) { return _mm256_sub_epi64(a, b); }
    static __m256 SubPs(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
    static __m256d SubPd(__m256d a, __m256d b) { return _mm256_sub_pd(a, b); }
    
    static Vec Mul(Vec a, Vec b) { return _mm256_mullo_epi32(a, b); }
    static __m256 MulPs(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
    static __m256d MulPd(__m256d a, __m256d b) { return _mm256_mul_pd(a, b); }
    
    static Vec And(Vec a, Vec b) { return _mm256_and_si256(a, b); }
    static Vec Or(Vec a, Vec b) { return _mm256_or_si256(a, b); }
    static Vec Xor(Vec a, Vec b) { return _mm256_xor_si256(a, b); }
    static Vec Not(Vec a) { return _mm256_xor_si256(a, _mm256_set1_epi32(-1)); }
    
    static int PopCount(Mask m) { return __builtin_popcount(m); }
};

template<>
struct SimdOps<SimdLevel::SSE42> {
    static constexpr size_t Width = 16;
    using Vec = __m128i;
    using Mask = int;
    
    static Vec Load(const void* ptr) { return _mm_loadu_si128(static_cast<const __m128i*>(ptr)); }
    static void Store(void* ptr, Vec v) { _mm_storeu_si128(static_cast<__m128i*>(ptr), v); }
    static Vec Set1(int32_t v) { return _mm_set1_epi32(v); }
    static Vec Set1(int64_t v) { return _mm_set1_epi64x(v); }
    
    static Mask CmpEq(Vec a, Vec b) { return _mm_movemask_epi8(_mm_cmpeq_epi32(a, b)); }
    static Mask CmpLt(Vec a, Vec b) { return _mm_movemask_epi8(_mm_cmpgt_epi32(b, a)); }
    
    static Vec Add(Vec a, Vec b) { return _mm_add_epi32(a, b); }
    static Vec Sub(Vec a, Vec b) { return _mm_sub_epi32(a, b); }
    static Vec Mul(Vec a, Vec b) { return _mm_mullo_epi32(a, b); }
    static Vec And(Vec a, Vec b) { return _mm_and_si128(a, b); }
    static Vec Or(Vec a, Vec b) { return _mm_or_si128(a, b); }
    static Vec Xor(Vec a, Vec b) { return _mm_xor_si128(a, b); }
    static Vec Not(Vec a) { return _mm_xor_si128(a, _mm_set1_epi32(-1)); }
};

template<>
struct SimdOps<SimdLevel::Scalar> {
    static constexpr size_t Width = 1;
};

// Runtime SIMD dispatch helper - use GetBestSimdLevel() at runtime
// instead of this compile-time alias (GetBestSimdLevel is not constexpr)

inline void Prefetch(const void* ptr, int locality = 3) {
    _mm_prefetch(static_cast<const char*>(ptr), static_cast<_mm_hint>(locality));
}

inline void PrefetchNTA(const void* ptr) {
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_NTA);
}

inline void SFence() { _mm_sfence(); }
inline void MFence() { _mm_mfence(); }
inline void LFence() { _mm_lfence(); }

} // namespace simd
} // namespace columnar
