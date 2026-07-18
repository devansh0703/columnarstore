#pragma once

#include <columnar/types.h>
#include <columnar/encoding/encoding.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

namespace columnar {
namespace rle {

template<DataType T>
struct RlePair {
    typename TypeTraits<T>::Type value;
    uint32_t count;
    
    RlePair() : value{}, count(0) {}
    RlePair(typename TypeTraits<T>::Type v, uint32_t c) : value(v), count(c) {}
};

template<DataType T>
class RleEncoder {
    using ValueType = typename TypeTraits<T>::Type;
    
public:
    std::vector<RlePair<T>> runs_;
    
    RleEncoder() = default;
    
    void Encode(const ValueType* data, size_t count) {
        if (count == 0) return;
        
        runs_.clear();
        runs_.reserve(count / 4 + 1);
        
        ValueType current = data[0];
        uint32_t run_length = 1;
        
        for (size_t i = 1; i < count; ++i) {
            if (data[i] == current && run_length < UINT32_MAX) {
                ++run_length;
            } else {
                runs_.emplace_back(current, run_length);
                current = data[i];
                run_length = 1;
            }
        }
        runs_.emplace_back(current, run_length);
    }
    
    void Write(encoding::Buffer& out) const {
        out.Append(static_cast<uint32_t>(runs_.size()));
        for (const auto& run : runs_) {
            out.Append(run.value);
            out.Append(run.count);
        }
    }
    
    size_t EstimateSize() const {
        return sizeof(uint32_t) + runs_.size() * (sizeof(ValueType) + sizeof(uint32_t));
    }
    
    static RleEncoder Read(const void* data, size_t size) {
        RleEncoder enc;
        const uint8_t* ptr = static_cast<const uint8_t*>(data);
        
        uint32_t num_runs;
        std::memcpy(&num_runs, ptr, 4);
        ptr += 4;
        
        enc.runs_.resize(num_runs);
        for (uint32_t i = 0; i < num_runs; ++i) {
            std::memcpy(&enc.runs_[i].value, ptr, sizeof(ValueType));
            ptr += sizeof(ValueType);
            std::memcpy(&enc.runs_[i].count, ptr, 4);
            ptr += 4;
        }
        
        return enc;
    }
};

template<DataType T>
class RleDecoder {
    using ValueType = typename TypeTraits<T>::Type;
    
    const RlePair<T>* runs_;
    size_t num_runs_;
    
public:
    RleDecoder() : runs_(nullptr), num_runs_(0) {}
    RleDecoder(const RlePair<T>* runs, size_t num_runs) : runs_(runs), num_runs_(num_runs) {}
    
    void SetData(const RlePair<T>* runs, size_t num_runs) {
        runs_ = runs;
        num_runs_ = num_runs;
    }
    
    void Decode(ValueType* out, size_t count) const {
        size_t out_pos = 0;
        for (size_t i = 0; i < num_runs_ && out_pos < count; ++i) {
            size_t to_write = std::min<size_t>(runs_[i].count, count - out_pos);
            std::fill_n(out + out_pos, to_write, runs_[i].value);
            out_pos += to_write;
        }
    }
    
    void DecodeAvx512(ValueType* out, size_t count) const {
#if defined(__AVX512F__)
        size_t out_pos = 0;
        if constexpr (sizeof(ValueType) == 4) {
            for (size_t i = 0; i < num_runs_ && out_pos < count; ++i) {
                size_t to_write = std::min<size_t>(runs_[i].count, count - out_pos);
                ValueType value = runs_[i].value;
                
                __m512i v = _mm512_set1_epi32(value);
                size_t simd_writes = to_write & ~15;
                for (size_t j = 0; j < simd_writes; j += 16) {
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + out_pos + j), v);
                }
                for (size_t j = simd_writes; j < to_write; ++j) {
                    out[out_pos + j] = value;
                }
                out_pos += to_write;
            }
        } else if constexpr (sizeof(ValueType) == 8) {
            for (size_t i = 0; i < num_runs_ && out_pos < count; ++i) {
                size_t to_write = std::min<size_t>(runs_[i].count, count - out_pos);
                ValueType value = runs_[i].value;
                
                __m512i v = _mm512_set1_epi64(value);
                size_t simd_writes = to_write & ~7;
                for (size_t j = 0; j < simd_writes; j += 8) {
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + out_pos + j), v);
                }
                for (size_t j = simd_writes; j < to_write; ++j) {
                    out[out_pos + j] = value;
                }
                out_pos += to_write;
            }
        }
#else
        Decode(out, count);
#endif
    }
    
    void DecodeAvx2(ValueType* out, size_t count) const {
        size_t out_pos = 0;
        if constexpr (sizeof(ValueType) == 4) {
            for (size_t i = 0; i < num_runs_ && out_pos < count; ++i) {
                size_t to_write = std::min<size_t>(runs_[i].count, count - out_pos);
                ValueType value = runs_[i].value;
                
                __m256i v = _mm256_set1_epi32(value);
                size_t simd_writes = to_write & ~7;
                for (size_t j = 0; j < simd_writes; j += 8) {
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + out_pos + j), v);
                }
                for (size_t j = simd_writes; j < to_write; ++j) {
                    out[out_pos + j] = value;
                }
                out_pos += to_write;
            }
        } else {
            Decode(out, count);
        }
    }
    
    template<PredicateType Pred>
    void Filter(const ValueType* value, bool* mask, size_t count) const {
        size_t out_pos = 0;
        for (size_t i = 0; i < num_runs_ && out_pos < count; ++i) {
            size_t to_write = std::min<size_t>(runs_[i].count, count - out_pos);
            bool match = false;
            
            if constexpr (Pred == PredicateType::Equal) {
                match = (runs_[i].value == *value);
            } else if constexpr (Pred == PredicateType::NotEqual) {
                match = (runs_[i].value != *value);
            } else if constexpr (Pred == PredicateType::LessThan) {
                match = (runs_[i].value < *value);
            } else if constexpr (Pred == PredicateType::LessEqual) {
                match = (runs_[i].value <= *value);
            } else if constexpr (Pred == PredicateType::GreaterThan) {
                match = (runs_[i].value > *value);
            } else if constexpr (Pred == PredicateType::GreaterEqual) {
                match = (runs_[i].value >= *value);
            }
            
            if (match) {
                for (size_t j = 0; j < to_write; ++j) {
                    mask[out_pos + j] = true;
                }
            } else {
                for (size_t j = 0; j < to_write; ++j) {
                    mask[out_pos + j] = false;
                }
            }
            out_pos += to_write;
        }
    }
};

template<DataType T>
inline std::unique_ptr<encoding::Encoder> CreateRleEncoder() {
    return std::make_unique<encoding::RLEEncoder<T>>();
}

template<DataType T>
inline std::unique_ptr<encoding::Decoder> CreateRleDecoder() {
    return std::make_unique<encoding::RLEDecoder<T>>();
}

} // namespace rle
} // namespace columnar
