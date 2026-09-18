#pragma once

#include <columnar/types.h>
#include <columnar/encoding/encoding.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

namespace columnar {
namespace bitpack {

template<DataType T>
class BitPackEncoder {
    using ValueType = typename TypeTraits<T>::Type;
    using UnsignedType = typename std::make_unsigned<ValueType>::type;
    
public:
    struct BitPackedData {
        ValueType min_val;
        uint8_t bits;
        std::vector<uint64_t> packed;
        
        size_t Size() const { return packed.size() * sizeof(uint64_t); }
    };
    
    BitPackedData Encode(const ValueType* data, size_t count) {
        BitPackedData result;
        
        if (count == 0) {
            result.min_val = ValueType{};
            result.bits = 0;
            return result;
        }
        
        ValueType min_val = data[0];
        ValueType max_val = data[0];
        for (size_t i = 1; i < count; ++i) {
            min_val = std::min(min_val, data[i]);
            max_val = std::max(max_val, data[i]);
        }
        
        result.min_val = min_val;
        
        UnsignedType range = static_cast<UnsignedType>(max_val) - static_cast<UnsignedType>(min_val);
        result.bits = 0;
        while ((UnsignedType(1) << result.bits) <= range && result.bits < 64) ++result.bits;
        
        if (result.bits == 0) {
            return result;
        }
        
        size_t packed_words = (count * result.bits + 63) / 64;
        result.packed.resize(packed_words, 0);
        
        size_t bit_pos = 0;
        UnsignedType mask = (result.bits == 64) ? ~0ULL : ((UnsignedType(1) << result.bits) - 1);
        
        for (size_t i = 0; i < count; ++i) {
            UnsignedType value = static_cast<UnsignedType>(data[i]) - static_cast<UnsignedType>(min_val);
            value &= mask;
            
            size_t word = bit_pos / 64;
            size_t offset = bit_pos % 64;
            
            if (offset + result.bits <= 64) {
                result.packed[word] |= value << offset;
            } else {
                result.packed[word] |= value << offset;
                result.packed[word + 1] |= value >> (64 - offset);
            }
            bit_pos += result.bits;
        }
        
        return result;
    }
    
    void Write(const BitPackedData& data, encoding::Buffer& out) const {
        out.Append(data.min_val);
        out.Append(data.bits);
        out.Append(static_cast<uint32_t>(data.packed.size()));
        out.AppendArray(data.packed.data(), data.packed.size());
    }
    
    static BitPackedData Read(const uint8_t* data, size_t /*size*/) {
        BitPackedData result;
        const uint8_t* ptr = data;
        
        std::memcpy(&result.min_val, ptr, sizeof(ValueType));
        ptr += sizeof(ValueType);
        std::memcpy(&result.bits, ptr, 1);
        ptr += 1;
        
        uint32_t num_words;
        std::memcpy(&num_words, ptr, 4);
        ptr += 4;
        
        result.packed.resize(num_words);
        std::memcpy(result.packed.data(), ptr, num_words * sizeof(uint64_t));
        
        return result;
    }
};

template<DataType T>
class BitPackDecoder {
    using ValueType = typename TypeTraits<T>::Type;
    using UnsignedType = typename std::make_unsigned<ValueType>::type;
    
public:
    void Decode(const BitPackEncoder<T>::BitPackedData& data, ValueType* out, size_t count) {
        if (count == 0 || data.bits == 0) {
            std::fill_n(out, count, data.min_val);
            return;
        }
        
        const uint64_t* packed = data.packed.data();
        UnsignedType mask = (data.bits == 64) ? ~0ULL : ((UnsignedType(1) << data.bits) - 1);
        size_t bit_pos = 0;
        
        for (size_t i = 0; i < count; ++i) {
            size_t word = bit_pos / 64;
            size_t offset = bit_pos % 64;
            
            UnsignedType value;
            if (offset + data.bits <= 64) {
                value = (packed[word] >> offset) & mask;
            } else {
                value = ((packed[word] >> offset) | (packed[word + 1] << (64 - offset))) & mask;
            }
            
            out[i] = data.min_val + static_cast<ValueType>(value);
            bit_pos += data.bits;
        }
    }
    
    void DecodeAvx512(const BitPackEncoder<T>::BitPackedData& data, ValueType* out, size_t count) {
#ifdef __AVX512F__
        if constexpr (sizeof(ValueType) == 4) {
            DecodeAvx512_32(data, out, count);
        } else if constexpr (sizeof(ValueType) == 8) {
            DecodeAvx512_64(data, out, count);
        } else {
            Decode(data, out, count);
        }
#else
        Decode(data, out, count);
#endif
    }
    
    void DecodeAvx2(const BitPackEncoder<T>::BitPackedData& data, ValueType* out, size_t count) {
        if constexpr (sizeof(ValueType) == 4) {
            DecodeAvx2_32(data, out, count);
        } else {
            Decode(data, out, count);
        }
    }
    
private:
    void DecodeAvx512_32(const BitPackEncoder<T>::BitPackedData& data, int32_t* out, size_t count) {
#ifdef __AVX512F__
        if (count == 0 || data.bits == 0) {
            std::fill_n(out, count, data.min_val);
            return;
        }
        
        const uint64_t* packed = data.packed.data();
        uint64_t mask = (data.bits == 64) ? ~0ULL : ((1ULL << data.bits) - 1);
        
        size_t i = 0;
        for (; i + 15 < count; i += 16) {
            size_t bit_start = i * data.bits;
            size_t word_start = bit_start / 64;
            size_t offset = bit_start % 64;
            
            if (offset == 0 && data.bits == 32) {
                __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(packed + word_start));
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + i), _mm512_add_epi32(v, _mm512_set1_epi32(data.min_val)));
                continue;
            }
            
            for (size_t j = 0; j < 16; ++j) {
                size_t bit_pos = (i + j) * data.bits;
                size_t word = bit_pos / 64;
                size_t off = bit_pos % 64;
                
                uint64_t value;
                if (off + data.bits <= 64) {
                    value = (packed[word] >> off) & mask;
                } else {
                    value = ((packed[word] >> off) | (packed[word + 1] << (64 - off))) & mask;
                }
                out[i + j] = data.min_val + static_cast<int32_t>(value);
            }
        }
        
        for (; i < count; ++i) {
            size_t bit_pos = i * data.bits;
            size_t word = bit_pos / 64;
            size_t offset = bit_pos % 64;
            
            uint64_t value;
            if (offset + data.bits <= 64) {
                value = (packed[word] >> offset) & mask;
            } else {
                value = ((packed[word] >> offset) | (packed[word + 1] << (64 - offset))) & mask;
            }
            out[i] = data.min_val + static_cast<int32_t>(value);
        }
#else
        if (count == 0 || data.bits == 0) {
            std::fill_n(out, count, static_cast<int32_t>(data.min_val));
            return;
        }
        {
            const uint64_t* packed = data.packed.data();
            uint64_t mask = (data.bits == 64) ? ~0ULL : ((1ULL << data.bits) - 1);
            for (size_t i = 0; i < count; ++i) {
                size_t bit_pos = i * data.bits;
                size_t word = bit_pos / 64;
                size_t offset = bit_pos % 64;
                uint64_t value;
                if (offset + data.bits <= 64) {
                    value = (packed[word] >> offset) & mask;
                } else {
                    value = ((packed[word] >> offset) | (packed[word + 1] << (64 - offset))) & mask;
                }
                out[i] = static_cast<int32_t>(data.min_val) + static_cast<int32_t>(value);
            }
        }
#endif
    }
    
    void DecodeAvx2_32(const BitPackEncoder<T>::BitPackedData& data, int32_t* out, size_t count) {
        if (count == 0 || data.bits == 0) {
            std::fill_n(out, count, data.min_val);
            return;
        }
        
        const uint64_t* packed = data.packed.data();
        uint64_t mask = (data.bits == 64) ? ~0ULL : ((1ULL << data.bits) - 1);
        
        size_t i = 0;
        for (; i + 7 < count; i += 8) {
            for (size_t j = 0; j < 8; ++j) {
                size_t bit_pos = (i + j) * data.bits;
                size_t word = bit_pos / 64;
                size_t offset = bit_pos % 64;
                
                uint64_t value;
                if (offset + data.bits <= 64) {
                    value = (packed[word] >> offset) & mask;
                } else {
                    value = ((packed[word] >> offset) | (packed[word + 1] << (64 - offset))) & mask;
                }
                out[i + j] = data.min_val + static_cast<int32_t>(value);
            }
        }
        
        for (; i < count; ++i) {
            size_t bit_pos = i * data.bits;
            size_t word = bit_pos / 64;
            size_t offset = bit_pos % 64;
            
            uint64_t value;
            if (offset + data.bits <= 64) {
                value = (packed[word] >> offset) & mask;
            } else {
                value = ((packed[word] >> offset) | (packed[word + 1] << (64 - offset))) & mask;
            }
            out[i] = data.min_val + static_cast<int32_t>(value);
        }
    }
    
    void DecodeAvx512_64(const BitPackEncoder<T>::BitPackedData& data, int64_t* out, size_t count) {
#ifdef __AVX512F__
        if (count == 0 || data.bits == 0) {
            std::fill_n(out, count, data.min_val);
            return;
        }
        
        const uint64_t* packed = data.packed.data();
        uint64_t mask = (data.bits == 64) ? ~0ULL : ((1ULL << data.bits) - 1);
        
        for (size_t i = 0; i < count; ++i) {
            size_t bit_pos = i * data.bits;
            size_t word = bit_pos / 64;
            size_t offset = bit_pos % 64;
            
            uint64_t value;
            if (offset + data.bits <= 64) {
                value = (packed[word] >> offset) & mask;
            } else {
                value = ((packed[word] >> offset) | (packed[word + 1] << (64 - offset))) & mask;
            }
            out[i] = data.min_val + static_cast<int64_t>(value);
        }
#else
        if (count == 0 || data.bits == 0) {
            std::fill_n(out, count, static_cast<int64_t>(data.min_val));
            return;
        }
        {
            const uint64_t* packed = data.packed.data();
            uint64_t mask = (data.bits == 64) ? ~0ULL : ((1ULL << data.bits) - 1);
            for (size_t i = 0; i < count; ++i) {
                size_t bit_pos = i * data.bits;
                size_t word = bit_pos / 64;
                size_t offset = bit_pos % 64;
                uint64_t value;
                if (offset + data.bits <= 64) {
                    value = (packed[word] >> offset) & mask;
                } else {
                    value = ((packed[word] >> offset) | (packed[word + 1] << (64 - offset))) & mask;
                }
                out[i] = static_cast<int64_t>(data.min_val) + static_cast<int64_t>(value);
            }
        }
#endif
    }
};

template<DataType T>
inline std::unique_ptr<encoding::Encoder> CreateBitPackEncoder() {
    return std::make_unique<encoding::BitPackEncoder<T, 64>>();
}

template<DataType T>
inline std::unique_ptr<encoding::Decoder> CreateBitPackDecoder() {
    return std::make_unique<encoding::BitPackDecoder<T, 64>>();
}

} // namespace bitpack
} // namespace columnar
