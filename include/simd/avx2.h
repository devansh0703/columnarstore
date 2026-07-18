#pragma once

#include <simd/dispatch.h>
#include <immintrin.h>

namespace columnar {
namespace simd {
namespace avx2 {

using Vec256i = __m256i;
using Vec256f = __m256;
using Vec256d = __m256d;
using Mask = int;

inline Vec256i Load(const void* ptr) { return _mm256_loadu_si256(static_cast<const Vec256i*>(ptr)); }
inline void Store(void* ptr, Vec256i v) { _mm256_storeu_si256(static_cast<Vec256i*>(ptr), v); }
inline Vec256f LoadPs(const void* ptr) { return _mm256_loadu_ps(static_cast<const float*>(ptr)); }
inline void StorePs(void* ptr, Vec256f v) { _mm256_storeu_ps(static_cast<float*>(ptr), v); }
inline Vec256d LoadPd(const void* ptr) { return _mm256_loadu_pd(static_cast<const double*>(ptr)); }
inline void StorePd(void* ptr, Vec256d v) { _mm256_storeu_pd(static_cast<double*>(ptr), v); }

inline Vec256i Set1(int32_t v) { return _mm256_set1_epi32(v); }
inline Vec256i Set1(int64_t v) { return _mm256_set1_epi64x(v); }
inline Vec256f Set1(float v) { return _mm256_set1_ps(v); }
inline Vec256d Set1(double v) { return _mm256_set1_pd(v); }

inline Mask CmpEq(Vec256i a, Vec256i b) {
    return _mm256_movemask_epi8(_mm256_cmpeq_epi32(a, b));
}
inline Mask CmpEq64(Vec256i a, Vec256i b) {
    return _mm256_movemask_epi8(_mm256_cmpeq_epi64(a, b));
}
inline Mask CmpEqPs(Vec256f a, Vec256f b) {
    return _mm256_movemask_ps(_mm256_cmp_ps(a, b, _CMP_EQ_OQ));
}
inline Mask CmpEqPd(Vec256d a, Vec256d b) {
    return _mm256_movemask_pd(_mm256_cmp_pd(a, b, _CMP_EQ_OQ));
}

inline Mask CmpLt(Vec256i a, Vec256i b) {
    return _mm256_movemask_epi8(_mm256_cmpgt_epi32(b, a));
}
inline Mask CmpLt64(Vec256i a, Vec256i b) {
    return _mm256_movemask_epi8(_mm256_cmpgt_epi64(b, a));
}
inline Mask CmpLtPs(Vec256f a, Vec256f b) {
    return _mm256_movemask_ps(_mm256_cmp_ps(a, b, _CMP_LT_OQ));
}
inline Mask CmpLtPd(Vec256d a, Vec256d b) {
    return _mm256_movemask_pd(_mm256_cmp_pd(a, b, _CMP_LT_OQ));
}

inline Mask CmpLe(Vec256i a, Vec256i b) {
    return _mm256_movemask_epi8(_mm256_cmpgt_epi32(_mm256_sub_epi32(b, a), _mm256_setzero_si256()));
}
inline Mask CmpLePs(Vec256f a, Vec256f b) {
    return _mm256_movemask_ps(_mm256_cmp_ps(a, b, _CMP_LE_OQ));
}
inline Mask CmpLePd(Vec256d a, Vec256d b) {
    return _mm256_movemask_pd(_mm256_cmp_pd(a, b, _CMP_LE_OQ));
}

inline Vec256i Add(Vec256i a, Vec256i b) { return _mm256_add_epi32(a, b); }
inline Vec256i Add64(Vec256i a, Vec256i b) { return _mm256_add_epi64(a, b); }
inline Vec256f AddPs(Vec256f a, Vec256f b) { return _mm256_add_ps(a, b); }
inline Vec256d AddPd(Vec256d a, Vec256d b) { return _mm256_add_pd(a, b); }

inline Vec256i Sub(Vec256i a, Vec256i b) { return _mm256_sub_epi32(a, b); }
inline Vec256i Sub64(Vec256i a, Vec256i b) { return _mm256_sub_epi64(a, b); }
inline Vec256f SubPs(Vec256f a, Vec256f b) { return _mm256_sub_ps(a, b); }
inline Vec256d SubPd(Vec256d a, Vec256d b) { return _mm256_sub_pd(a, b); }

inline Vec256i Mul(Vec256i a, Vec256i b) { return _mm256_mullo_epi32(a, b); }
inline Vec256f MulPs(Vec256f a, Vec256f b) { return _mm256_mul_ps(a, b); }
inline Vec256d MulPd(Vec256d a, Vec256d b) { return _mm256_mul_pd(a, b); }

inline Vec256i And(Vec256i a, Vec256i b) { return _mm256_and_si256(a, b); }
inline Vec256i Or(Vec256i a, Vec256i b) { return _mm256_or_si256(a, b); }
inline Vec256i Xor(Vec256i a, Vec256i b) { return _mm256_xor_si256(a, b); }
inline Vec256i Not(Vec256i a) { return _mm256_xor_si256(a, _mm256_set1_epi32(-1)); }

inline Vec256i Shuffle(Vec256i a, int imm) { return _mm256_shuffle_epi32(a, imm); }
inline Vec256i Permute(Vec256i a, Vec256i idx) { return _mm256_permutevar8x32_epi32(a, idx); }
inline Vec256i Blend(Vec256i a, Vec256i b, int mask) { return _mm256_blend_epi32(a, b, mask); }
inline Vec256i Blendv(Vec256i a, Vec256i b, Vec256i mask) { return _mm256_blendv_epi8(a, b, mask); }

inline Vec256i Slli(Vec256i a, int shift) { return _mm256_slli_epi32(a, shift); }
inline Vec256i Srli(Vec256i a, int shift) { return _mm256_srli_epi32(a, shift); }
inline Vec256i Srai(Vec256i a, int shift) { return _mm256_srai_epi32(a, shift); }

inline Vec256i Packus(Vec256i a, Vec256i b) { return _mm256_packus_epi32(a, b); }
inline Vec256i Packss(Vec256i a, Vec256i b) { return _mm256_packs_epi32(a, b); }

inline int PopCount(Mask m) { return __builtin_popcount(m); }
inline int PopCount64(uint64_t m) { return __builtin_popcountll(m); }

inline Vec256i MaskLoad(const void* ptr, Mask m) {
    int32_t tmp[8];
    const int32_t* src = static_cast<const int32_t*>(ptr);
    for (int i = 0; i < 8; ++i) {
        tmp[i] = (m & (1u << i)) ? src[i] : 0;
    }
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tmp));
}
inline void MaskStore(void* ptr, Mask m, Vec256i v) {
    int32_t tmp[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), v);
    int32_t* dst = static_cast<int32_t*>(ptr);
    for (int i = 0; i < 8; ++i) {
        if (m & (1u << i)) dst[i] = tmp[i];
    }
}

inline Vec256i PrefixSum(Vec256i v) {
    v = _mm256_add_epi32(v, _mm256_slli_si256(v, 4));
    v = _mm256_add_epi32(v, _mm256_slli_si256(v, 8));
    v = _mm256_add_epi32(v, _mm256_slli_si256(v, 16));
    return v;
}

inline Vec256i ShuffleEpi8(Vec256i a, Vec256i idx) { return _mm256_shuffle_epi8(a, idx); }

} // namespace avx2
} // namespace simd
} // namespace columnar
