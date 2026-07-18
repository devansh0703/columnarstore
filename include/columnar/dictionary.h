#pragma once

#include <columnar/types.h>
#include <columnar/encoding/encoding.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <immintrin.h>

namespace columnar {
namespace dictionary {

template<DataType T>
class DictionaryEncoder {
    using ValueType = typename TypeTraits<T>::Type;
    
    std::vector<ValueType> dictionary_;
    std::vector<uint32_t> indices_;
    std::unordered_map<ValueType, uint32_t> value_to_id_;
    
public:
    DictionaryEncoder() = default;
    
    void Build(const ValueType* values, size_t count) {
        dictionary_.reserve(std::min(count, size_t(65536)));
        value_to_id_.reserve(std::min(count * 2, size_t(131072)));
        indices_.resize(count);
        
        for (size_t i = 0; i < count; ++i) {
            ValueType v = values[i];
            auto it = value_to_id_.find(v);
            if (it == value_to_id_.end()) {
                uint32_t id = static_cast<uint32_t>(dictionary_.size());
                value_to_id_[v] = id;
                dictionary_.push_back(v);
                indices_[i] = id;
            } else {
                indices_[i] = it->second;
            }
        }
    }
    
    void BuildSorted(const ValueType* values, size_t count) {
        std::vector<ValueType> sorted(values, values + count);
        std::sort(sorted.begin(), sorted.end());
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
        
        dictionary_ = std::move(sorted);
        value_to_id_.reserve(dictionary_.size() * 2);
        for (uint32_t i = 0; i < dictionary_.size(); ++i) {
            value_to_id_[dictionary_[i]] = i;
        }
        
        indices_.resize(count);
        for (size_t i = 0; i < count; ++i) {
            indices_[i] = value_to_id_[values[i]];
        }
    }
    
    const std::vector<ValueType>& Dictionary() const { return dictionary_; }
    const std::vector<uint32_t>& Indices() const { return indices_; }
    size_t DictionarySize() const { return dictionary_.size(); }
    size_t Cardinality() const { return dictionary_.size(); }
    
    void Encode(const ValueType* values, size_t count, uint32_t* out) const {
        for (size_t i = 0; i < count; ++i) {
            out[i] = value_to_id_.at(values[i]);
        }
    }
    
    void Decode(const uint32_t* indices, size_t count, ValueType* out) const {
        for (size_t i = 0; i < count; ++i) {
            out[i] = dictionary_[indices[i]];
        }
    }
    
    void DecodeAvx512(const uint32_t* indices, size_t count, ValueType* out) const {
#ifdef __AVX512F__
        size_t i = 0;
        if constexpr (sizeof(ValueType) == 4) {
            for (; i + 15 < count; i += 16) {
                __m512i idx = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(indices + i));
                __m512i result = _mm512_i32gather_epi32(idx, reinterpret_cast<const int*>(dictionary_.data()), 4);
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + i), result);
            }
        } else if constexpr (sizeof(ValueType) == 8) {
            for (; i + 7 < count; i += 8) {
                __m512i idx = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(indices + i));
                __m512i result = _mm512_i64gather_epi64(idx, reinterpret_cast<const long long*>(dictionary_.data()), 8);
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + i), result);
            }
        }
        for (; i < count; ++i) {
            out[i] = dictionary_[indices[i]];
        }
#else
        Decode(indices, count, out);
#endif
    }
    
    void DecodeAvx2(const uint32_t* indices, size_t count, ValueType* out) const {
        size_t i = 0;
        if constexpr (sizeof(ValueType) == 4) {
            for (; i + 7 < count; i += 8) {
                __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices + i));
                __m256i result = _mm256_i32gather_epi32(reinterpret_cast<const int*>(dictionary_.data()), idx, 4);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), result);
            }
        } else if constexpr (sizeof(ValueType) == 8) {
            for (; i + 3 < count; i += 4) {
                __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices + i));
                __m256i result = _mm256_i64gather_epi64(reinterpret_cast<const long long*>(dictionary_.data()), idx, 8);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), result);
            }
        }
        for (; i < count; ++i) {
            out[i] = dictionary_[indices[i]];
        }
    }
    
    size_t EstimateSize() const {
        return dictionary_.size() * sizeof(ValueType) + indices_.size() * sizeof(uint32_t);
    }
    
    void Write(encoding::Buffer& out) const {
        out.Append(static_cast<uint32_t>(dictionary_.size()));
        out.Append(dictionary_.data(), dictionary_.size() * sizeof(ValueType));
        
        uint8_t index_bits = 0;
        uint32_t dict_size = static_cast<uint32_t>(dictionary_.size());
        while ((1U << index_bits) < dict_size && index_bits < 32) ++index_bits;
        out.Append(index_bits);
        
        if (index_bits <= 8) {
            std::vector<uint8_t> idx8(indices_.size());
            for (size_t i = 0; i < indices_.size(); ++i) idx8[i] = static_cast<uint8_t>(indices_[i]);
            out.Append(idx8.data(), idx8.size());
        } else if (index_bits <= 16) {
            std::vector<uint16_t> idx16(indices_.size());
            for (size_t i = 0; i < indices_.size(); ++i) idx16[i] = static_cast<uint16_t>(indices_[i]);
            out.Append(idx16.data(), idx16.size() * 2);
        } else {
            out.Append(indices_.data(), indices_.size() * 4);
        }
    }
    
    static DictionaryEncoder Read(const uint8_t* data, size_t size) {
        DictionaryEncoder enc;
        const uint8_t* ptr = data;
        
        uint32_t dict_size;
        std::memcpy(&dict_size, ptr, 4);
        ptr += 4;
        
        enc.dictionary_.resize(dict_size);
        std::memcpy(enc.dictionary_.data(), ptr, dict_size * sizeof(ValueType));
        ptr += dict_size * sizeof(ValueType);
        
        enc.value_to_id_.reserve(dict_size * 2);
        for (uint32_t i = 0; i < dict_size; ++i) {
            enc.value_to_id_[enc.dictionary_[i]] = i;
        }
        
        uint8_t index_bits;
        std::memcpy(&index_bits, ptr, 1);
        ptr += 1;
        
        size_t remaining = size - (ptr - data);
        size_t count = remaining / ((index_bits <= 8) ? 1 : (index_bits <= 16 ? 2 : 4));
        
        enc.indices_.resize(count);
        if (index_bits <= 8) {
            const uint8_t* idx = reinterpret_cast<const uint8_t*>(ptr);
            for (size_t i = 0; i < count; ++i) enc.indices_[i] = idx[i];
        } else if (index_bits <= 16) {
            const uint16_t* idx = reinterpret_cast<const uint16_t*>(ptr);
            for (size_t i = 0; i < count; ++i) enc.indices_[i] = idx[i];
        } else {
            const uint32_t* idx = reinterpret_cast<const uint32_t*>(ptr);
            for (size_t i = 0; i < count; ++i) enc.indices_[i] = idx[i];
        }
        
        return enc;
    }
};

template<DataType T>
class DictionaryDecoder {
    std::vector<typename TypeTraits<T>::Type> dictionary_;
    
public:
    DictionaryDecoder() = default;
    explicit DictionaryDecoder(const std::vector<typename TypeTraits<T>::Type>& dict) : dictionary_(dict) {}
    
    const std::vector<typename TypeTraits<T>::Type>& Dictionary() const { return dictionary_; }
    
    void Decode(const uint32_t* indices, size_t count, typename TypeTraits<T>::Type* out) const {
        for (size_t i = 0; i < count; ++i) {
            out[i] = dictionary_[indices[i]];
        }
    }
    
    void DecodeAvx512(const uint32_t* indices, size_t count, typename TypeTraits<T>::Type* out) const {
#ifdef __AVX512F__
        size_t i = 0;
        if constexpr (sizeof(typename TypeTraits<T>::Type) == 4) {
            for (; i + 15 < count; i += 16) {
                __m512i idx = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(indices + i));
                __m512i result = _mm512_i32gather_epi32(idx, reinterpret_cast<const int*>(dictionary_.data()), 4);
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + i), result);
            }
        } else if constexpr (sizeof(typename TypeTraits<T>::Type) == 8) {
            for (; i + 7 < count; i += 8) {
                __m512i idx = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(indices + i));
                __m512i result = _mm512_i64gather_epi64(idx, reinterpret_cast<const long long*>(dictionary_.data()), 8);
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + i), result);
            }
        }
        for (; i < count; ++i) out[i] = dictionary_[indices[i]];
#else
        Decode(indices, count, out);
#endif
    }
    
    void DecodeAvx2(const uint32_t* indices, size_t count, typename TypeTraits<T>::Type* out) const {
        size_t i = 0;
        if constexpr (sizeof(typename TypeTraits<T>::Type) == 4) {
            for (; i + 7 < count; i += 8) {
                __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices + i));
                __m256i result = _mm256_i32gather_epi32(reinterpret_cast<const int*>(dictionary_.data()), idx, 4);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), result);
            }
        } else if constexpr (sizeof(typename TypeTraits<T>::Type) == 8) {
            for (; i + 3 < count; i += 4) {
                __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices + i));
                __m256i result = _mm256_i64gather_epi64(reinterpret_cast<const long long*>(dictionary_.data()), idx, 8);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), result);
            }
        }
        for (; i < count; ++i) out[i] = dictionary_[indices[i]];
    }
};

template<DataType T>
inline std::unique_ptr<encoding::Encoder> CreateDictionaryEncoder() {
    return std::make_unique<encoding::DictEncoder<T>>();
}

template<DataType T>
inline std::unique_ptr<encoding::Decoder> CreateDictionaryDecoder() {
    return std::make_unique<encoding::DictDecoder<T>>();
}

} // namespace dictionary
} // namespace columnar