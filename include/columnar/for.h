#pragma once

#include <columnar/types.h>
#include <columnar/encoding/encoding.h>
#include <columnar/bitpack.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

namespace columnar {
namespace for_encoding {

template<DataType T>
class ForEncoder {
    using ValueType = typename TypeTraits<T>::Type;
    using UnsignedType = typename std::make_unsigned<ValueType>::type;
    
public:
    struct ForHeader {
        ValueType base;
        uint8_t bits;
        uint32_t count;
    };
    
    void Encode(const ValueType* data, size_t count, encoding::Buffer& out) {
        if (count == 0) return;
        
        ValueType min_val = data[0];
        for (size_t i = 1; i < count; ++i) {
            if (data[i] < min_val) min_val = data[i];
        }
        
        ValueType max_val = data[0];
        for (size_t i = 1; i < count; ++i) {
            if (data[i] > max_val) max_val = data[i];
        }
        
        UnsignedType range = static_cast<UnsignedType>(max_val) - static_cast<UnsignedType>(min_val);
        uint8_t bits = 0;
        while ((UnsignedType(1) << bits) <= range && bits < 64) ++bits;
        
        out.Append(min_val);
        out.Append(bits);
        out.Append(static_cast<uint32_t>(count));
        
        if (bits == 0) return;
        
        size_t packed_words = (count * bits + 63) / 64;
        std::vector<uint64_t> packed(packed_words, 0);
        
        size_t bit_pos = 0;
        for (size_t i = 0; i < count; ++i) {
            UnsignedType delta = static_cast<UnsignedType>(data[i]) - static_cast<UnsignedType>(min_val);
            size_t word = bit_pos / 64;
            size_t offset = bit_pos % 64;
            
            if (offset + bits <= 64) {
                packed[word] |= delta << offset;
            } else {
                packed[word] |= delta << offset;
                packed[word + 1] |= delta >> (64 - offset);
            }
            bit_pos += bits;
        }
        
        out.Append(packed.data(), packed.size() * sizeof(uint64_t));
    }
    
    size_t EstimateSize(const ValueType* data, size_t count) const {
        if (count == 0) return sizeof(ForHeader);
        
        ValueType min_val = data[0];
        ValueType max_val = data[0];
        for (size_t i = 1; i < count; ++i) {
            min_val = std::min(min_val, data[i]);
            max_val = std::max(max_val, data[i]);
        }
        
        UnsignedType range = static_cast<UnsignedType>(max_val) - static_cast<UnsignedType>(min_val);
        uint8_t bits = 0;
        while ((UnsignedType(1) << bits) <= range && bits < 64) ++bits;
        
        return sizeof(ForHeader) + ((count * bits + 63) / 64) * 8;
    }
};

template<DataType T>
class ForDecoder {
    using ValueType = typename TypeTraits<T>::Type;
    using UnsignedType = typename std::make_unsigned<ValueType>::type;
    
public:
    void Decode(const void* data, size_t /*size*/, ValueType* out, size_t count) {
        if (count == 0) return;
        
        const uint8_t* src = static_cast<const uint8_t*>(data);
        ValueType base;
        uint8_t bits;
        uint32_t stored_count;
        
        std::memcpy(&base, src, sizeof(ValueType));
        src += sizeof(ValueType);
        std::memcpy(&bits, src, 1);
        src += 1;
        std::memcpy(&stored_count, src, 4);
        src += 4;
        
        if (bits == 0) {
            std::fill_n(out, count, base);
            return;
        }
        
        const uint64_t* packed = reinterpret_cast<const uint64_t*>(src);
        UnsignedType mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);
        
        size_t bit_pos = 0;
        for (size_t i = 0; i < count; ++i) {
            size_t word = bit_pos / 64;
            size_t offset = bit_pos % 64;
            
            UnsignedType delta;
            if (offset + bits <= 64) {
                delta = (packed[word] >> offset) & mask;
            } else {
                delta = ((packed[word] >> offset) | (packed[word + 1] << (64 - offset))) & mask;
            }
            
            out[i] = base + static_cast<ValueType>(delta);
            bit_pos += bits;
        }
    }
    
    void DecodeAvx512(const void* data, size_t size, ValueType* out, size_t count) {
#ifdef __AVX512F__
        if constexpr (sizeof(ValueType) == 4) {
            DecodeAvx512_32(data, size, out, count);
        } else {
            Decode(data, size, out, count);
        }
#else
        Decode(data, size, out, count);
#endif
    }
    
    void DecodeAvx2(const void* data, size_t size, ValueType* out, size_t count) {
        if constexpr (sizeof(ValueType) == 4) {
            DecodeAvx2_32(data, size, out, count);
        } else {
            Decode(data, size, out, count);
        }
    }
    
private:
    void DecodeAvx512_32(const void* data, size_t /*size*/, int32_t* out, size_t count) {
#ifdef __AVX512F__
        const uint8_t* src = static_cast<const uint8_t*>(data);
        int32_t base;
        uint8_t bits;
        uint32_t stored_count;
        
        std::memcpy(&base, src, 4);
        src += 4;
        std::memcpy(&bits, src, 1);
        src += 1;
        std::memcpy(&stored_count, src, 4);
        src += 4;
        
        if (bits == 0) {
            std::fill_n(out, count, base);
            return;
        }
        
        const uint64_t* packed = reinterpret_cast<const uint64_t*>(src);
        uint64_t mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);
        
        size_t i = 0;
        for (; i + 15 < count; i += 16) {
            size_t bit_start = i * bits;
            size_t word_start = bit_start / 64;
            size_t offset = bit_start % 64;
            
            __m512i result;
            if (offset + 16 * bits <= 512 && bits <= 32) {
                __m512i load = _mm512_loadu_si512(packed + word_start);
                
                if (bits == 1) {
                    __m512i mask = _mm512_set1_epi32(1);
                    __m512i shifts = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
                    __m512i shifted = _mm512_srlv_epi32(_mm512_broadcast_i32x4(_mm512_castsi512_si128(load)), shifts);
                    __m512i extracted = _mm512_and_epi32(shifted, mask);
                    result = _mm512_add_epi32(extracted, _mm512_set1_epi32(base));
                } else if (bits <= 8) {
                    __m512i shifts = _mm512_set_epi32(15*bits,14*bits,13*bits,12*bits,11*bits,10*bits,9*bits,8*bits,
                                                       7*bits,6*bits,5*bits,4*bits,3*bits,2*bits,1*bits,0);
                    __m512i shifted = _mm512_srlv_epi32(_mm512_broadcast_i32x4(_mm512_castsi512_si128(load)), shifts);
                    __m512i extracted = _mm512_and_epi32(shifted, _mm512_set1_epi32(static_cast<int32_t>(mask)));
                    result = _mm512_add_epi32(extracted, _mm512_set1_epi32(base));
                } else if (bits <= 16) {
                    __m512i shifts = _mm512_set_epi32(15*bits,14*bits,13*bits,12*bits,11*bits,10*bits,9*bits,8*bits,
                                                       7*bits,6*bits,5*bits,4*bits,3*bits,2*bits,1*bits,0);
                    __m512i shifted = _mm512_srlv_epi32(_mm512_broadcast_i32x4(_mm512_castsi512_si128(load)), shifts);
                    __m512i extracted = _mm512_and_epi32(shifted, _mm512_set1_epi32(static_cast<int32_t>(mask)));
                    result = _mm512_add_epi32(extracted, _mm512_set1_epi32(base));
                } else {
                    for (size_t j = 0; j < 16; ++j) {
                        size_t bit_pos = (i + j) * bits;
                        size_t word = bit_pos / 64;
                        size_t off = bit_pos % 64;
                        uint64_t delta;
                        if (off + bits <= 64) {
                            delta = (packed[word] >> off) & mask;
                        } else {
                            delta = ((packed[word] >> off) | (packed[word + 1] << (64 - off))) & mask;
                        }
                        out[i + j] = base + static_cast<int32_t>(delta);
                    }
                    continue;
                }
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(out + i), result);
            } else {
                for (size_t j = 0; j < 16; ++j) {
                    size_t bit_pos = (i + j) * bits;
                    size_t word = bit_pos / 64;
                    size_t off = bit_pos % 64;
                    uint64_t delta;
                    if (off + bits <= 64) {
                        delta = (packed[word] >> off) & mask;
                    } else {
                        delta = ((packed[word] >> off) | (packed[word + 1] << (64 - off))) & mask;
                    }
                    out[i + j] = base + static_cast<int32_t>(delta);
                }
            }
        }
        
        for (; i < count; ++i) {
            size_t bit_pos = i * bits;
            size_t word = bit_pos / 64;
            size_t offset = bit_pos % 64;
            
            uint64_t delta;
            if (offset + bits <= 64) {
                delta = (packed[word] >> offset) & mask;
            } else {
                delta = ((packed[word] >> offset) | (packed[word + 1] << (64 - offset))) & mask;
            }
            out[i] = base + static_cast<int32_t>(delta);
        }
#else
        {
            const uint8_t* src = static_cast<const uint8_t*>(data);
            int32_t base;
            uint8_t bits;
            uint32_t stored_count;
            std::memcpy(&base, src, 4); src += 4;
            std::memcpy(&bits, src, 1); src += 1;
            std::memcpy(&stored_count, src, 4); src += 4;
            if (bits == 0) { std::fill_n(out, count, base); return; }
            const uint64_t* packed = reinterpret_cast<const uint64_t*>(src);
            uint64_t mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);
            for (size_t i = 0; i < count; ++i) {
                size_t bit_pos = i * bits;
                size_t word = bit_pos / 64;
                size_t offset = bit_pos % 64;
                uint64_t delta;
                if (offset + bits <= 64) {
                    delta = (packed[word] >> offset) & mask;
                } else {
                    delta = ((packed[word] >> offset) | (packed[word + 1] << (64 - offset))) & mask;
                }
                out[i] = base + static_cast<int32_t>(delta);
            }
        }
#endif
    }
    
    void DecodeAvx2_32(const void* data, size_t /*size*/, int32_t* out, size_t count) {
        const uint8_t* src = static_cast<const uint8_t*>(data);
        int32_t base;
        uint8_t bits;
        uint32_t stored_count;
        
        std::memcpy(&base, src, 4);
        src += 4;
        std::memcpy(&bits, src, 1);
        src += 1;
        std::memcpy(&stored_count, src, 4);
        src += 4;
        
        if (bits == 0) {
            std::fill_n(out, count, base);
            return;
        }
        
        const uint64_t* packed = reinterpret_cast<const uint64_t*>(src);
        uint64_t mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);
        
        for (size_t i = 0; i < count; ++i) {
            size_t bit_pos = i * bits;
            size_t word = bit_pos / 64;
            size_t offset = bit_pos % 64;
            
            uint64_t delta;
            if (offset + bits <= 64) {
                delta = (packed[word] >> offset) & mask;
            } else {
                delta = ((packed[word] >> offset) | (packed[word + 1] << (64 - offset))) & mask;
            }
            out[i] = base + static_cast<int32_t>(delta);
        }
    }
};

template<DataType T>
class ZigZagEncoder {
    using ValueType = typename TypeTraits<T>::Type;
    using UnsignedType = typename std::make_unsigned<ValueType>::type;
    
public:
    static UnsignedType Encode(ValueType value) {
        return (static_cast<UnsignedType>(value) << 1) ^ (static_cast<UnsignedType>(value) >> (sizeof(ValueType) * 8 - 1));
    }
    
    static ValueType Decode(UnsignedType value) {
        return static_cast<ValueType>((value >> 1) ^ -(value & 1));
    }
};

template<DataType T>
class ForZigZagEncoder : public ForEncoder<T> {
    using ValueType = typename TypeTraits<T>::Type;
    using UnsignedType = typename std::make_unsigned<ValueType>::type;
    
public:
    void Encode(const ValueType* data, size_t count, encoding::Buffer& out) {
        if (count == 0) return;
        
        UnsignedType min_zz = ZigZagEncoder<T>::Encode(data[0]);
        UnsignedType max_zz = min_zz;
        
        for (size_t i = 1; i < count; ++i) {
            UnsignedType zz = ZigZagEncoder<T>::Encode(data[i]);
            if (zz < min_zz) min_zz = zz;
            if (zz > max_zz) max_zz = zz;
        }
        
        ValueType min_val = ZigZagEncoder<T>::Decode(min_zz);
        UnsignedType range = max_zz - min_zz;
        uint8_t bits = 0;
        while ((UnsignedType(1) << bits) <= range && bits < 64) ++bits;
        
        out.Append(min_val);
        out.Append(bits);
        out.Append(static_cast<uint32_t>(count));
        
        if (bits == 0) return;
        
        size_t packed_words = (count * bits + 63) / 64;
        std::vector<uint64_t> packed(packed_words, 0);
        
        size_t bit_pos = 0;
        for (size_t i = 0; i < count; ++i) {
            UnsignedType zz = ZigZagEncoder<T>::Encode(data[i]);
            UnsignedType delta = zz - min_zz;
            
            size_t word = bit_pos / 64;
            size_t offset = bit_pos % 64;
            
            if (offset + bits <= 64) {
                packed[word] |= delta << offset;
            } else {
                packed[word] |= delta << offset;
                packed[word + 1] |= delta >> (64 - offset);
            }
            bit_pos += bits;
        }
        
        out.Append(packed.data(), packed.size() * sizeof(uint64_t));
    }
};

template<DataType T>
class ForZigZagDecoder : public ForDecoder<T> {
    using ValueType = typename TypeTraits<T>::Type;
    using UnsignedType = typename std::make_unsigned<ValueType>::type;
    
public:
    void Decode(const void* data, size_t /*size*/, ValueType* out, size_t count) {
        if (count == 0) return;
        
        const uint8_t* src = static_cast<const uint8_t*>(data);
        ValueType base;
        uint8_t bits;
        uint32_t stored_count;
        
        std::memcpy(&base, src, sizeof(ValueType));
        src += sizeof(ValueType);
        std::memcpy(&bits, src, 1);
        src += 1;
        std::memcpy(&stored_count, src, 4);
        src += 4;
        
        if (bits == 0) {
            std::fill_n(out, count, base);
            return;
        }
        
        const uint64_t* packed = reinterpret_cast<const uint64_t*>(src);
        UnsignedType mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);
        
        size_t bit_pos = 0;
        for (size_t i = 0; i < count; ++i) {
            size_t word = bit_pos / 64;
            size_t offset = bit_pos % 64;
            
            UnsignedType delta;
            if (offset + bits <= 64) {
                delta = (packed[word] >> offset) & mask;
            } else {
                delta = ((packed[word] >> offset) | (packed[word + 1] << (64 - offset))) & mask;
            }
            
            UnsignedType zz = ZigZagEncoder<T>::Decode(base) + delta;
            out[i] = ZigZagEncoder<T>::Decode(zz);
            bit_pos += bits;
        }
    }
};

} // namespace for_encoding
} // namespace columnar
