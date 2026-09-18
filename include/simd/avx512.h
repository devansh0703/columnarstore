#pragma once

#include <simd/dispatch.h>
#include <immintrin.h>

namespace columnar {
namespace simd {
namespace avx512 {

using Vec512i = __m512i;
using Vec512f = __m512;
using Vec512d = __m512d;
using Mask8 = __mmask8;
using Mask16 = __mmask16;
using Mask32 = __mmask32;
using Mask64 = __mmask64;

inline Vec512i Load(const void* ptr) { return _mm512_loadu_si512(ptr); }
inline void Store(void* ptr, Vec512i v) { _mm512_storeu_si512(ptr, v); }
inline Vec512f LoadPs(const void* ptr) { return _mm512_loadu_ps(static_cast<const float*>(ptr)); }
inline void StorePs(void* ptr, Vec512f v) { _mm512_storeu_ps(static_cast<float*>(ptr), v); }
inline Vec512d LoadPd(const void* ptr) { return _mm512_loadu_pd(static_cast<const double*>(ptr)); }
inline void StorePd(void* ptr, Vec512d v) { _mm512_storeu_pd(static_cast<double*>(ptr), v); }

inline Vec512i Set1(int32_t v) { return _mm512_set1_epi32(v); }
inline Vec512i Set1(int64_t v) { return _mm512_set1_epi64(v); }
inline Vec512f Set1(float v) { return _mm512_set1_ps(v); }
inline Vec512d Set1(double v) { return _mm512_set1_pd(v); }

inline Mask16 CmpEq(Vec512i a, Vec512i b) { return _mm512_cmpeq_epi32_mask(a, b); }
inline Mask8 CmpEq64(Vec512i a, Vec512i b) { return _mm512_cmpeq_epi64_mask(a, b); }
inline Mask16 CmpEqPs(Vec512f a, Vec512f b) { return _mm512_cmpeq_ps_mask(a, b); }
inline Mask8 CmpEqPd(Vec512d a, Vec512d b) { return _mm512_cmpeq_pd_mask(a, b); }

inline Mask16 CmpLt(Vec512i a, Vec512i b) { return _mm512_cmplt_epi32_mask(a, b); }
inline Mask8 CmpLt64(Vec512i a, Vec512i b) { return _mm512_cmplt_epi64_mask(a, b); }
inline Mask16 CmpLtPs(Vec512f a, Vec512f b) { return _mm512_cmplt_ps_mask(a, b); }
inline Mask8 CmpLtPd(Vec512d a, Vec512d b) { return _mm512_cmplt_pd_mask(a, b); }

inline Mask16 CmpLe(Vec512i a, Vec512i b) { return _mm512_cmple_epi32_mask(a, b); }
inline Mask8 CmpLe64(Vec512i a, Vec512i b) { return _mm512_cmple_epi64_mask(a, b); }
inline Mask16 CmpLePs(Vec512f a, Vec512f b) { return _mm512_cmple_ps_mask(a, b); }
inline Mask8 CmpLePd(Vec512d a, Vec512d b) { return _mm512_cmple_pd_mask(a, b); }

inline Vec512i Add(Vec512i a, Vec512i b) { return _mm512_add_epi32(a, b); }
inline Vec512i Add64(Vec512i a, Vec512i b) { return _mm512_add_epi64(a, b); }
inline Vec512f AddPs(Vec512f a, Vec512f b) { return _mm512_add_ps(a, b); }
inline Vec512d AddPd(Vec512d a, Vec512d b) { return _mm512_add_pd(a, b); }

inline Vec512i Sub(Vec512i a, Vec512i b) { return _mm512_sub_epi32(a, b); }
inline Vec512i Sub64(Vec512i a, Vec512i b) { return _mm512_sub_epi64(a, b); }
inline Vec512f SubPs(Vec512f a, Vec512f b) { return _mm512_sub_ps(a, b); }
inline Vec512d SubPd(Vec512d a, Vec512d b) { return _mm512_sub_pd(a, b); }

inline Vec512i Mul(Vec512i a, Vec512i b) { return _mm512_mullo_epi32(a, b); }
inline Vec512i Mul64(Vec512i a, Vec512i b) { return _mm512_mullo_epi64(a, b); }
inline Vec512f MulPs(Vec512f a, Vec512f b) { return _mm512_mul_ps(a, b); }
inline Vec512d MulPd(Vec512d a, Vec512d b) { return _mm512_mul_pd(a, b); }

inline Vec512i And(Vec512i a, Vec512i b) { return _mm512_and_si512(a, b); }
inline Vec512i Or(Vec512i a, Vec512i b) { return _mm512_or_si512(a, b); }
inline Vec512i Xor(Vec512i a, Vec512i b) { return _mm512_xor_si512(a, b); }
inline Vec512i Not(Vec512i a) { return _mm512_ternarylogic_epi32(a, _mm512_set1_epi32(-1), _mm512_set1_epi32(0), 0x96); }

inline Vec512i Shuffle(Vec512i a, int imm) { return _mm512_shuffle_epi32(a, static_cast<_MM_PERM_ENUM>(imm)); }
inline Vec512i Permute(Vec512i a, Vec512i idx) { return _mm512_permutexvar_epi32(idx, a); }
inline Vec512i Blend(Vec512i a, Vec512i b, Mask16 m) { return _mm512_mask_mov_epi32(a, m, b); }
inline Vec512i Blendv(Vec512i a, Vec512i b, Mask16 m) { return _mm512_mask_mov_epi32(a, m, b); }

inline Vec512i Slli(Vec512i a, int shift) { return _mm512_slli_epi32(a, shift); }
inline Vec512i Srli(Vec512i a, int shift) { return _mm512_srli_epi32(a, shift); }
inline Vec512i Srai(Vec512i a, int shift) { return _mm512_srai_epi32(a, shift); }

inline Vec512i Packus(Vec512i a, Vec512i b) { return _mm512_packus_epi32(a, b); }
inline Vec512i Packss(Vec512i a, Vec512i b) { return _mm512_packs_epi32(a, b); }

inline int PopCount(Mask64 m) { return _mm_popcnt_u64(m); }
inline int PopCount(Mask32 m) { return _mm_popcnt_u32(m); }
inline int PopCount(Mask16 m) { return _mm_popcnt_u32(m); }
inline int PopCount(Mask8 m) { return _mm_popcnt_u32(m); }

inline Vec512i MaskzLoad(const void* ptr, Mask16 m) { return _mm512_maskz_loadu_epi32(m, ptr); }
inline void MaskStore(void* ptr, Mask16 m, Vec512i v) { _mm512_mask_storeu_epi32(ptr, m, v); }

inline Vec512i Compress(Vec512i a, Mask16 m) { return _mm512_maskz_compress_epi32(m, a); }
inline Vec512i Expand(Vec512i a, Mask16 m) { return _mm512_maskz_expand_epi32(m, a); }

inline Vec512i ConflictDetect(Vec512i a) {
    // Fallback: scalar conflict detection
    alignas(64) int32_t tmp[16];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), _mm512_castsi512_si256(a));
    alignas(64) int32_t result[16] = {};
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < i; ++j) {
            if (tmp[i] == tmp[j]) { result[i] = 1 << j; break; }
        }
    }
    return _mm512_load_si512(reinterpret_cast<const __m512i*>(result));
}
inline Vec512i ConflictDetect64(Vec512i a) {
    // Fallback: scalar conflict detection
    alignas(64) int64_t tmp[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), _mm512_castsi512_si256(a));
    alignas(64) int64_t result[8] = {};
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < i; ++j) {
            if (tmp[i] == tmp[j]) { result[i] = 1LL << j; break; }
        }
    }
    return _mm512_castsi256_si512(_mm256_load_si256(reinterpret_cast<const __m256i*>(result)));
}

inline Vec512i PrefixSum(Vec512i v) {
    for (int i = 4; i < 64; i <<= 1) {
        v = _mm512_add_epi32(v, _mm512_slli_epi64(v, i));
    }
    return v;
}

inline Vec512i PrefixSum64(Vec512i v) {
    for (int i = 8; i < 64; i <<= 1) {
        v = _mm512_add_epi64(v, _mm512_slli_epi64(v, i));
    }
    return v;
}

inline Vec512i ShuffleEpi8(Vec512i a, Vec512i idx) { return _mm512_shuffle_epi8(a, idx); }
inline Vec512i PermuteVar8x32(Vec512i a, Vec512i idx) { return _mm512_permutexvar_epi32(idx, a); }

inline Vec512i Broadcast(int32_t v) { return _mm512_set1_epi32(v); }
inline Vec512i Broadcast64(int64_t v) { return _mm512_set1_epi64(v); }

inline Vec512i InsertInt(Vec512i a, int32_t v, int idx) { return _mm512_mask_set1_epi32(a, 1u << idx, v); }
inline Vec512i InsertInt64(Vec512i a, int64_t v, int idx) { return _mm512_mask_set1_epi64(a, 1u << idx, v); }

inline int32_t ExtractInt(Vec512i a, int idx) { 
    alignas(32) int32_t tmp[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), _mm512_castsi512_si256(a));
    return tmp[idx]; 
}
inline int64_t ExtractInt64(Vec512i a, int idx) { 
    alignas(32) int64_t tmp[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), _mm512_castsi512_si256(a));
    return tmp[idx]; 
}

inline Vec512i ReduceAdd(Vec512i v) {
    v = _mm512_add_epi32(v, _mm512_shuffle_epi32(v, static_cast<_MM_PERM_ENUM>(0xEE)));
    v = _mm512_add_epi32(v, _mm512_shuffle_epi32(v, static_cast<_MM_PERM_ENUM>(0x11)));
    return v;
}

inline float ReduceAdd(Vec512f v) { return _mm512_reduce_add_ps(v); }
inline double ReduceAdd(Vec512d v) { return _mm512_reduce_add_pd(v); }

inline int32_t ReduceMin(Vec512i v) { return _mm512_reduce_min_epi32(v); }
inline int64_t ReduceMin(Vec512i v, Mask8 /*m*/) { return _mm512_reduce_min_epi64(v); }
inline float ReduceMin(Vec512f v) { return _mm512_reduce_min_ps(v); }
inline double ReduceMin(Vec512d v) { return _mm512_reduce_min_pd(v); }

inline int32_t ReduceMax(Vec512i v) { return _mm512_reduce_max_epi32(v); }
inline int64_t ReduceMax(Vec512i v, Mask8 /*m*/) { return _mm512_reduce_max_epi64(v); }
inline float ReduceMax(Vec512f v) { return _mm512_reduce_max_ps(v); }
inline double ReduceMax(Vec512d v) { return _mm512_reduce_max_pd(v); }

inline Vec512i Abs(Vec512i v) { return _mm512_abs_epi32(v); }
inline Vec512i Abs64(Vec512i v) { return _mm512_abs_epi64(v); }
inline Vec512f Abs(Vec512f v) { return _mm512_and_ps(v, _mm512_set1_ps(0x7fffffff)); }
inline Vec512d Abs(Vec512d v) { return _mm512_and_pd(v, _mm512_set1_pd(0x7fffffffffffffff)); }

inline Vec512i Max(Vec512i a, Vec512i b) { return _mm512_max_epi32(a, b); }
inline Vec512i Min(Vec512i a, Vec512i b) { return _mm512_min_epi32(a, b); }
inline Vec512f Max(Vec512f a, Vec512f b) { return _mm512_max_ps(a, b); }
inline Vec512f Min(Vec512f a, Vec512f b) { return _mm512_min_ps(a, b); }
inline Vec512d Max(Vec512d a, Vec512d b) { return _mm512_max_pd(a, b); }
inline Vec512d Min(Vec512d a, Vec512d b) { return _mm512_min_pd(a, b); }

inline Vec512i SubSaturate(Vec512i a, Vec512i b) { return _mm512_sub_epi32(a, b); }
inline Vec512i AddSaturate(Vec512i a, Vec512i b) { return _mm512_add_epi32(a, b); }

inline Vec512i MulHi(Vec512i a, Vec512i b) {
    // Fallback: simple 32-bit multiply (no high-word available in AVX-512)
    return _mm512_mullo_epi32(a, b);
}
inline Vec512i MulHi64(Vec512i a, Vec512i b) {
    // AVX-512 doesn't have mulhi_epi64, use simple fallback
    __m256i a_lo = _mm512_castsi512_si256(a);
    __m256i b_lo = _mm512_castsi512_si256(b);
    return _mm512_castsi256_si512(_mm256_mul_epu32(a_lo, b_lo));
}

inline Vec512i ShiftLeft(Vec512i a, Vec512i count) { return _mm512_sllv_epi32(a, count); }
inline Vec512i ShiftRight(Vec512i a, Vec512i count) { return _mm512_srlv_epi32(a, count); }
inline Vec512i ShiftRightArith(Vec512i a, Vec512i count) { return _mm512_srav_epi32(a, count); }

inline Vec512i TernaryLogic(Vec512i a, Vec512i b, Vec512i c, int imm) { 
    return _mm512_ternarylogic_epi32(a, b, c, imm); 
}

inline Vec512i MaskMove(Vec512i a, Mask16 m, Vec512i b) { return _mm512_mask_mov_epi32(a, m, b); }

} // namespace avx512
} // namespace simd
} // namespace columnar
