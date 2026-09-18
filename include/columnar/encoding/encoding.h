#pragma once

#include <columnar/types.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <memory>
#include <variant>
#include <string_view>
#include <type_traits>

namespace columnar {
namespace encoding {

class Buffer {
public:
    Buffer() = default;
    explicit Buffer(size_t capacity) : data_(capacity), size_(0) {}
    Buffer(const void* data, size_t size) : data_(size), size_(size) {
        std::memcpy(data_.data(), data, size);
    }
    
    void Reserve(size_t capacity) { data_.reserve(capacity); }
    void Resize(size_t size) { size_ = size; data_.resize(size); }
    void Append(const void* data, size_t size) {
        size_t old = size_;
        size_ += size;
        data_.resize(size_);
        std::memcpy(data_.data() + old, data, size);
    }
    
    template<typename T>
    void Append(const T& value) {
        Append(&value, sizeof(T));
    }
    
    void Append(const unsigned char* data, size_t count) {
        Append(static_cast<const void*>(data), count * sizeof(unsigned char));
    }
    
    void Append(const char* data, size_t count) {
        Append(static_cast<const void*>(data), count * sizeof(char));
    }
    
    void AppendArray(const uint16_t* data, size_t count) {
        Append(data, count * sizeof(uint16_t));
    }
    
    void AppendArray(const int16_t* data, size_t count) {
        Append(data, count * sizeof(int16_t));
    }
    
    void AppendArray(const uint32_t* data, size_t count) {
        Append(data, count * sizeof(uint32_t));
    }
    
    void AppendArray(const int32_t* data, size_t count) {
        Append(data, count * sizeof(int32_t));
    }
    
    void AppendArray(const uint64_t* data, size_t count) {
        Append(data, count * sizeof(uint64_t));
    }
    
    void AppendArray(const int64_t* data, size_t count) {
        Append(data, count * sizeof(int64_t));
    }
    
    void AppendArray(const float* data, size_t count) {
        Append(data, count * sizeof(float));
    }
    
    void AppendArray(const double* data, size_t count) {
        Append(data, count * sizeof(double));
    }
    
    void* Data() { return data_.data(); }
    const void* Data() const { return data_.data(); }
    template<typename T>
    T* DataAs() { return reinterpret_cast<T*>(data_.data()); }
    template<typename T>
    const T* DataAs() const { return reinterpret_cast<const T*>(data_.data()); }
    
    size_t Size() const { return size_; }
    size_t Capacity() const { return data_.size(); }
    void Clear() { size_ = 0; }
    
private:
    std::vector<uint8_t> data_;
    size_t size_ = 0;
};

class Encoder {
public:
    virtual ~Encoder() = default;
    virtual EncodingType Type() const = 0;
    virtual void Encode(const void* data, size_t count, Buffer& out) = 0;
    virtual void Finish(Buffer&) {}
};

class Decoder {
public:
    virtual ~Decoder() = default;
    virtual EncodingType Type() const = 0;
    virtual void Decode(const void* data, size_t /*size*/, void* out, size_t count) = 0;
};

template<DataType T>
class PlainEncoder : public Encoder {
public:
    EncodingType Type() const override { return EncodingType::Plain; }
    
    void Encode(const void* data, size_t count, Buffer& out) override {
        out.AppendArray(static_cast<const NativeType<T>*>(data), count);
    }
};

template<DataType T>
class PlainDecoder : public Decoder {
public:
    EncodingType Type() const override { return EncodingType::Plain; }
    
    void Decode(const void* data, size_t /*size*/, void* out, size_t count) override {
        std::memcpy(out, data, count * sizeof(NativeType<T>));
    }
};

template<DataType T>
class RLEEncoder : public Encoder {
public:
    EncodingType Type() const override { return EncodingType::RLE; }
    
    void Encode(const void* data, size_t count, Buffer& out) override {
        const NativeType<T>* src = static_cast<const NativeType<T>*>(data);
        if (count == 0) return;
        
        NativeType<T> current = src[0];
        uint32_t run_length = 1;
        
        for (size_t i = 1; i < count; ++i) {
            if (src[i] == current && run_length < UINT32_MAX) {
                ++run_length;
            } else {
                out.Append(current);
                out.Append(run_length);
                current = src[i];
                run_length = 1;
            }
        }
        out.Append(current);
        out.Append(run_length);
    }
};

template<DataType T>
class RLEDecoder : public Decoder {
public:
    EncodingType Type() const override { return EncodingType::RLE; }
    
    void Decode(const void* data, size_t size, void* out, size_t count) override {
        const uint8_t* src = static_cast<const uint8_t*>(data);
        NativeType<T>* dst = static_cast<NativeType<T>*>(out);
        size_t src_offset = 0;
        size_t dst_offset = 0;
        
        while (dst_offset < count && src_offset + sizeof(NativeType<T>) + sizeof(uint32_t) <= size) {
            NativeType<T> value;
            uint32_t run_length;
            std::memcpy(&value, src + src_offset, sizeof(NativeType<T>));
            src_offset += sizeof(NativeType<T>);
            std::memcpy(&run_length, src + src_offset, sizeof(uint32_t));
            src_offset += sizeof(uint32_t);
            
            size_t to_write = std::min<size_t>(run_length, count - dst_offset);
            std::fill_n(dst + dst_offset, to_write, value);
            dst_offset += to_write;
        }
    }
};

template<DataType T>
class DeltaEncoder : public Encoder {
public:
    EncodingType Type() const override { return EncodingType::Delta; }
    
    void Encode(const void* data, size_t count, Buffer& out) override {
        if (count == 0) return;
        const NativeType<T>* src = static_cast<const NativeType<T>*>(data);
        NativeType<T> prev = src[0];
        out.Append(prev);
        
        for (size_t i = 1; i < count; ++i) {
            NativeType<T> delta = src[i] - prev;
            prev = src[i];
            out.Append(delta);
        }
    }
};

template<DataType T>
class DeltaDecoder : public Decoder {
public:
    EncodingType Type() const override { return EncodingType::Delta; }
    
    void Decode(const void* data, size_t /*size*/, void* out, size_t count) override {
        if (count == 0) return;
        const NativeType<T>* src = static_cast<const NativeType<T>*>(data);
        NativeType<T>* dst = static_cast<NativeType<T>*>(out);
        
        dst[0] = src[0];
        for (size_t i = 1; i < count; ++i) {
            dst[i] = dst[i - 1] + src[i];
        }
    }
};

template<DataType T, int BITS>
class BitPackEncoder : public Encoder {
    static_assert(BITS > 0 && BITS <= 64);
    using Unsigned = typename std::conditional_t<(BITS <= 8), uint8_t,
                        typename std::conditional_t<(BITS <= 16), uint16_t,
                        typename std::conditional_t<(BITS <= 32), uint32_t, uint64_t>>>;
    
public:
    EncodingType Type() const override { return EncodingType::BitPack; }
    
    void Encode(const void* data, size_t count, Buffer& out) override {
        const NativeType<T>* src = static_cast<const NativeType<T>*>(data);
        if (count == 0) return;
        
        NativeType<T> min_val = src[0];
        NativeType<T> max_val = src[0];
        for (size_t i = 1; i < count; ++i) {
            min_val = std::min(min_val, src[i]);
            max_val = std::max(max_val, src[i]);
        }
        
        out.Append(min_val);
        out.Append(max_val);
        
        size_t bits_needed = 0;
        if constexpr (std::is_signed_v<NativeType<T>>) {
            int64_t range = static_cast<int64_t>(max_val) - static_cast<int64_t>(min_val);
            while ((1ULL << bits_needed) <= static_cast<uint64_t>(range)) ++bits_needed;
        } else {
            uint64_t range = static_cast<uint64_t>(max_val) - static_cast<uint64_t>(min_val);
            while ((1ULL << bits_needed) <= range) ++bits_needed;
        }
        bits_needed = std::min<size_t>(bits_needed, BITS);
        
        out.Append(static_cast<uint8_t>(bits_needed));
        
        std::vector<Unsigned> packed((count * bits_needed + sizeof(Unsigned) * 8 - 1) / (sizeof(Unsigned) * 8));
        size_t bit_pos = 0;
        
        for (size_t i = 0; i < count; ++i) {
            uint64_t value = static_cast<uint64_t>(src[i] - min_val);
            size_t word = bit_pos / (sizeof(Unsigned) * 8);
            size_t offset = bit_pos % (sizeof(Unsigned) * 8);
            
            if (offset + bits_needed <= sizeof(Unsigned) * 8) {
                packed[word] |= static_cast<Unsigned>(value) << offset;
            } else {
                packed[word] |= static_cast<Unsigned>(value) << offset;
                packed[word + 1] |= static_cast<Unsigned>(value) >> (sizeof(Unsigned) * 8 - offset);
            }
            bit_pos += bits_needed;
        }
        
        out.AppendArray(packed.data(), packed.size());
    }
    
private:
    size_t bits_needed = 0;
};

template<DataType T, int BITS>
class BitPackDecoder : public Decoder {
    static_assert(BITS > 0 && BITS <= 64);
    using Unsigned = typename std::conditional_t<(BITS <= 8), uint8_t,
                        typename std::conditional_t<(BITS <= 16), uint16_t,
                        typename std::conditional_t<(BITS <= 32), uint32_t, uint64_t>>>;
    
public:
    EncodingType Type() const override { return EncodingType::BitPack; }
    
    void Decode(const void* data, size_t /*size*/, void* out, size_t count) override {
        if (count == 0) return;
        const uint8_t* src = static_cast<const uint8_t*>(data);
        NativeType<T>* dst = static_cast<NativeType<T>*>(out);
        
        NativeType<T> min_val;
        NativeType<T> max_val;
        std::memcpy(&min_val, src, sizeof(NativeType<T>));
        src += sizeof(NativeType<T>);
        std::memcpy(&max_val, src, sizeof(NativeType<T>));
        src += sizeof(NativeType<T>);
        
        uint8_t bits;
        std::memcpy(&bits, src, 1);
        src += 1;
        
        size_t bit_pos = 0;
        for (size_t i = 0; i < count; ++i) {
            uint64_t value = 0;
            size_t word = bit_pos / (sizeof(Unsigned) * 8);
            size_t offset = bit_pos % (sizeof(Unsigned) * 8);
            
            const Unsigned* packed = reinterpret_cast<const Unsigned*>(src);
            if (offset + bits <= sizeof(Unsigned) * 8) {
                value = (packed[word] >> offset) & ((1ULL << bits) - 1);
            } else {
                value = (packed[word] >> offset) | 
                       (static_cast<uint64_t>(packed[word + 1]) << (sizeof(Unsigned) * 8 - offset));
                value &= ((1ULL << bits) - 1);
            }
            
            dst[i] = min_val + static_cast<NativeType<T>>(value);
            bit_pos += bits;
        }
    }
};

template<DataType T>
class FOREncoder : public Encoder {
public:
    EncodingType Type() const override { return EncodingType::FOR; }
    
    void Encode(const void* data, size_t count, Buffer& out) override {
        const NativeType<T>* src = static_cast<const NativeType<T>*>(data);
        if (count == 0) return;
        
        NativeType<T> min_val = src[0];
        NativeType<T> max_val = src[0];
        for (size_t i = 1; i < count; ++i) {
            min_val = std::min(min_val, src[i]);
            max_val = std::max(max_val, src[i]);
        }
        
        out.Append(min_val);
        out.Append(max_val);
        
        uint8_t bits = 0;
        if constexpr (std::is_signed_v<NativeType<T>>) {
            int64_t range = static_cast<int64_t>(max_val) - static_cast<int64_t>(min_val);
            while ((1ULL << bits) <= static_cast<uint64_t>(range)) ++bits;
        } else {
            uint64_t range = static_cast<uint64_t>(max_val) - static_cast<uint64_t>(min_val);
            while ((1ULL << bits) <= range) ++bits;
        }
        out.Append(bits);
        
        if (bits == 0) return;
        
        size_t values_per_word = 64 / bits;
        size_t words = (count + values_per_word - 1) / values_per_word;
        std::vector<uint64_t> packed(words, 0);
        
        for (size_t i = 0; i < count; ++i) {
            uint64_t value = static_cast<uint64_t>(src[i] - min_val);
            size_t word = i / values_per_word;
            size_t shift = (i % values_per_word) * bits;
            packed[word] |= value << shift;
        }
        
        out.AppendArray(packed.data(), words);
    }
};

template<DataType T>
class FORDecoder : public Decoder {
public:
    EncodingType Type() const override { return EncodingType::FOR; }
    
    void Decode(const void* data, size_t /*size*/, void* out, size_t count) override {
        if (count == 0) return;
        const uint8_t* src = static_cast<const uint8_t*>(data);
        NativeType<T>* dst = static_cast<NativeType<T>*>(out);
        
        NativeType<T> min_val, max_val;
        std::memcpy(&min_val, src, sizeof(NativeType<T>));
        src += sizeof(NativeType<T>);
        std::memcpy(&max_val, src, sizeof(NativeType<T>));
        src += sizeof(NativeType<T>);
        
        uint8_t bits;
        std::memcpy(&bits, src, 1);
        src += 1;
        
        if (bits == 0) {
            std::fill_n(dst, count, min_val);
            return;
        }
        
        const uint64_t* packed = reinterpret_cast<const uint64_t*>(src);
        size_t values_per_word = 64 / bits;
        uint64_t mask = (1ULL << bits) - 1;
        
        for (size_t i = 0; i < count; ++i) {
            size_t word = i / values_per_word;
            size_t shift = (i % values_per_word) * bits;
            uint64_t value = (packed[word] >> shift) & mask;
            dst[i] = min_val + static_cast<NativeType<T>>(value);
        }
    }
};

template<DataType T>
class DictEncoder : public Encoder {
public:
    EncodingType Type() const override { return EncodingType::Dictionary; }
    
    void Encode(const void* data, size_t count, Buffer& out) override {
        const NativeType<T>* src = static_cast<const NativeType<T>*>(data);
        if (count == 0) return;
        
        std::vector<NativeType<T>> dict(src, src + count);
        std::sort(dict.begin(), dict.end());
        dict.erase(std::unique(dict.begin(), dict.end()), dict.end());
        
        out.Append(static_cast<uint32_t>(dict.size()));
        out.AppendArray(dict.data(), dict.size());
        
        uint32_t dict_size = static_cast<uint32_t>(dict.size());
        uint8_t index_bits = 0;
        while ((1U << index_bits) < dict_size && index_bits < 32) ++index_bits;
        out.Append(index_bits);
        
        if (index_bits <= 8) {
            std::vector<uint8_t> indices(count);
            for (size_t i = 0; i < count; ++i) {
                indices[i] = static_cast<uint8_t>(
                    std::lower_bound(dict.begin(), dict.end(), src[i]) - dict.begin());
            }
            out.Append(indices.data(), count);
        } else if (index_bits <= 16) {
            std::vector<uint16_t> indices(count);
            for (size_t i = 0; i < count; ++i) {
                indices[i] = static_cast<uint16_t>(
                    std::lower_bound(dict.begin(), dict.end(), src[i]) - dict.begin());
            }
            out.Append(indices.data(), count);
        } else {
            std::vector<uint32_t> indices(count);
            for (size_t i = 0; i < count; ++i) {
                indices[i] = static_cast<uint32_t>(
                    std::lower_bound(dict.begin(), dict.end(), src[i]) - dict.begin());
            }
            out.Append(indices.data(), count);
        }
    }
};

template<DataType T>
class DictDecoder : public Decoder {
public:
    EncodingType Type() const override { return EncodingType::Dictionary; }
    
    void Decode(const void* data, size_t /*size*/, void* out, size_t count) override {
        if (count == 0) return;
        const uint8_t* src = static_cast<const uint8_t*>(data);
        NativeType<T>* dst = static_cast<NativeType<T>*>(out);
        
        uint32_t dict_size;
        std::memcpy(&dict_size, src, 4);
        src += 4;
        
        const NativeType<T>* dict = reinterpret_cast<const NativeType<T>*>(src);
        src += dict_size * sizeof(NativeType<T>);
        
        uint8_t index_bits;
        std::memcpy(&index_bits, src, 1);
        src += 1;
        
        if (index_bits == 0) {
            std::fill_n(dst, count, dict[0]);
            return;
        }
        
        if (index_bits <= 8) {
            const uint8_t* indices = reinterpret_cast<const uint8_t*>(src);
            for (size_t i = 0; i < count; ++i) dst[i] = dict[indices[i]];
        } else if (index_bits <= 16) {
            const uint16_t* indices = reinterpret_cast<const uint16_t*>(src);
            for (size_t i = 0; i < count; ++i) dst[i] = dict[indices[i]];
        } else {
            const uint32_t* indices = reinterpret_cast<const uint32_t*>(src);
            for (size_t i = 0; i < count; ++i) dst[i] = dict[indices[i]];
        }
    }
};

template<EncodingType Type, DataType T>
std::unique_ptr<Encoder> CreateEncoder();

template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Plain, DataType::Int32>() {
    return std::make_unique<PlainEncoder<DataType::Int32>>();
}
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Plain, DataType::Int64>() {
    return std::make_unique<PlainEncoder<DataType::Int64>>();
}
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Plain, DataType::Float>() {
    return std::make_unique<PlainEncoder<DataType::Float>>();
}
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Plain, DataType::Double>() {
    return std::make_unique<PlainEncoder<DataType::Double>>();
}

template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::RLE, DataType::Int32>() {
    return std::make_unique<RLEEncoder<DataType::Int32>>();
}
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::RLE, DataType::Int64>() {
    return std::make_unique<RLEEncoder<DataType::Int64>>();
}

template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Delta, DataType::Int32>() {
    return std::make_unique<DeltaEncoder<DataType::Int32>>();
}
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Delta, DataType::Int64>() {
    return std::make_unique<DeltaEncoder<DataType::Int64>>();
}

template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::FOR, DataType::Int32>() {
    return std::make_unique<FOREncoder<DataType::Int32>>();
}
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::FOR, DataType::Int64>() {
    return std::make_unique<FOREncoder<DataType::Int64>>();
}


template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Delta, DataType::Varchar>() { return nullptr; }
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Plain, DataType::Varchar>() { return nullptr; }
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Dictionary, DataType::Varchar>() { return nullptr; }
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::RLE, DataType::Varchar>() { return nullptr; }
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::BitPack, DataType::Varchar>() { return nullptr; }
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::FOR, DataType::Varchar>() { return nullptr; }

template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Dictionary, DataType::Int32>() {
    return std::make_unique<DictEncoder<DataType::Int32>>();
}
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Dictionary, DataType::Int64>() {
    return std::make_unique<DictEncoder<DataType::Int64>>();
}

template<EncodingType Type, DataType T>
std::unique_ptr<Decoder> CreateDecoder();

template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Delta, DataType::Varchar>() { return nullptr; }
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Plain, DataType::Varchar>() { return nullptr; }
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Dictionary, DataType::Varchar>() { return nullptr; }
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::RLE, DataType::Varchar>() { return nullptr; }
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::BitPack, DataType::Varchar>() { return nullptr; }
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::FOR, DataType::Varchar>() { return nullptr; }

template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Plain, DataType::Int32>() {
    return std::make_unique<PlainDecoder<DataType::Int32>>();
}
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Plain, DataType::Int64>() {
    return std::make_unique<PlainDecoder<DataType::Int64>>();
}
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Plain, DataType::Float>() {
    return std::make_unique<PlainDecoder<DataType::Float>>();
}
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Plain, DataType::Double>() {
    return std::make_unique<PlainDecoder<DataType::Double>>();
}

template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::RLE, DataType::Int32>() {
    return std::make_unique<RLEDecoder<DataType::Int32>>();
}
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::RLE, DataType::Int64>() {
    return std::make_unique<RLEDecoder<DataType::Int64>>();
}

template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Delta, DataType::Int32>() {
    return std::make_unique<DeltaDecoder<DataType::Int32>>();
}
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Delta, DataType::Int64>() {
    return std::make_unique<DeltaDecoder<DataType::Int64>>();
}

template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::FOR, DataType::Int32>() {
    return std::make_unique<FORDecoder<DataType::Int32>>();
}
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::FOR, DataType::Int64>() {
    return std::make_unique<FORDecoder<DataType::Int64>>();
}

template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Dictionary, DataType::Int32>() {
    return std::make_unique<DictDecoder<DataType::Int32>>();
}
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Dictionary, DataType::Int64>() {
    return std::make_unique<DictDecoder<DataType::Int64>>();
}

// BitPack encoder/decoder specializations (using default 32-bit width)

// BitPack + Float/Double specializations (using Float as 32-bit, Double as 64-bit)
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::BitPack, DataType::Float>() {
    return std::make_unique<BitPackEncoder<DataType::Float, 32>>();
}
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::BitPack, DataType::Double>() {
    return std::make_unique<BitPackEncoder<DataType::Double, 64>>();
}
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::BitPack, DataType::Float>() {
    return std::make_unique<BitPackDecoder<DataType::Float, 32>>();
}
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::BitPack, DataType::Double>() {
    return std::make_unique<BitPackDecoder<DataType::Double, 64>>();
}

// BitPack encoder/decoder specializations
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::BitPack, DataType::Int32>() {
    return std::make_unique<BitPackEncoder<DataType::Int32, 32>>();
}
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::BitPack, DataType::Int64>() {
    return std::make_unique<BitPackEncoder<DataType::Int64, 64>>();
}
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::BitPack, DataType::Int32>() {
    return std::make_unique<BitPackDecoder<DataType::Int32, 32>>();
}
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::BitPack, DataType::Int64>() {
    return std::make_unique<BitPackDecoder<DataType::Int64, 64>>();
}

// Float/Double specializations for all encoding types
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::RLE, DataType::Float>() {
    return std::make_unique<RLEEncoder<DataType::Float>>();
}
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::RLE, DataType::Double>() {
    return std::make_unique<RLEEncoder<DataType::Double>>();
}
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Delta, DataType::Float>() {
    return std::make_unique<DeltaEncoder<DataType::Float>>();
}
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Delta, DataType::Double>() {
    return std::make_unique<DeltaEncoder<DataType::Double>>();
}
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::FOR, DataType::Float>() {
    return std::make_unique<FOREncoder<DataType::Float>>();
}
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::FOR, DataType::Double>() {
    return std::make_unique<FOREncoder<DataType::Double>>();
}
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Dictionary, DataType::Float>() {
    return std::make_unique<DictEncoder<DataType::Float>>();
}
template<> inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Dictionary, DataType::Double>() {
    return std::make_unique<DictEncoder<DataType::Double>>();
}
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::RLE, DataType::Float>() {
    return std::make_unique<RLEDecoder<DataType::Float>>();
}
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::RLE, DataType::Double>() {
    return std::make_unique<RLEDecoder<DataType::Double>>();
}
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Delta, DataType::Float>() {
    return std::make_unique<DeltaDecoder<DataType::Float>>();
}
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Delta, DataType::Double>() {
    return std::make_unique<DeltaDecoder<DataType::Double>>();
}
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::FOR, DataType::Float>() {
    return std::make_unique<FORDecoder<DataType::Float>>();
}
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::FOR, DataType::Double>() {
    return std::make_unique<FORDecoder<DataType::Double>>();
}
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Dictionary, DataType::Float>() {
    return std::make_unique<DictDecoder<DataType::Float>>();
}
template<> inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Dictionary, DataType::Double>() {
    return std::make_unique<DictDecoder<DataType::Double>>();
}

inline std::unique_ptr<Encoder> CreateEncoder(EncodingType enc, DataType dt) {
    switch (enc) {
        case EncodingType::Plain:
            switch (dt) {
                case DataType::Int32: return CreateEncoder<EncodingType::Plain, DataType::Int32>();
                case DataType::Int64: return CreateEncoder<EncodingType::Plain, DataType::Int64>();
                case DataType::Float: return CreateEncoder<EncodingType::Plain, DataType::Float>();
                case DataType::Double: return CreateEncoder<EncodingType::Plain, DataType::Double>();
                default: return CreateEncoder<EncodingType::Plain, DataType::Int32>();
            }
        case EncodingType::RLE:
            switch (dt) {
                case DataType::Int32: return CreateEncoder<EncodingType::RLE, DataType::Int32>();
                case DataType::Int64: return CreateEncoder<EncodingType::RLE, DataType::Int64>();
                case DataType::Float: return CreateEncoder<EncodingType::RLE, DataType::Float>();
                case DataType::Double: return CreateEncoder<EncodingType::RLE, DataType::Double>();
                default: return CreateEncoder<EncodingType::Plain, DataType::Int32>();
            }
        case EncodingType::Delta:
            switch (dt) {
                case DataType::Int32: return CreateEncoder<EncodingType::Delta, DataType::Int32>();
                case DataType::Int64: return CreateEncoder<EncodingType::Delta, DataType::Int64>();
                default: return CreateEncoder<EncodingType::Plain, DataType::Int32>();
            }
        case EncodingType::FOR:
            switch (dt) {
                case DataType::Int32: return CreateEncoder<EncodingType::FOR, DataType::Int32>();
                case DataType::Int64: return CreateEncoder<EncodingType::FOR, DataType::Int64>();
                default: return CreateEncoder<EncodingType::Plain, DataType::Int32>();
            }
        case EncodingType::Dictionary:
            switch (dt) {
                case DataType::Int32: return CreateEncoder<EncodingType::Dictionary, DataType::Int32>();
                case DataType::Int64: return CreateEncoder<EncodingType::Dictionary, DataType::Int64>();
                case DataType::Float: return CreateEncoder<EncodingType::Dictionary, DataType::Float>();
                case DataType::Double: return CreateEncoder<EncodingType::Dictionary, DataType::Double>();
                default: return CreateEncoder<EncodingType::Plain, DataType::Int32>();
            }
        case EncodingType::BitPack:
            switch (dt) {
                case DataType::Int32: return CreateEncoder<EncodingType::BitPack, DataType::Int32>();
                case DataType::Int64: return CreateEncoder<EncodingType::BitPack, DataType::Int64>();
                default: return CreateEncoder<EncodingType::Plain, DataType::Int32>();
            }
        default: return CreateEncoder<EncodingType::Plain, DataType::Int32>();
    }
}

inline std::unique_ptr<Decoder> CreateDecoder(EncodingType enc, DataType dt) {
    switch (enc) {
        case EncodingType::Plain:
            switch (dt) {
                case DataType::Int32: return CreateDecoder<EncodingType::Plain, DataType::Int32>();
                case DataType::Int64: return CreateDecoder<EncodingType::Plain, DataType::Int64>();
                case DataType::Float: return CreateDecoder<EncodingType::Plain, DataType::Float>();
                case DataType::Double: return CreateDecoder<EncodingType::Plain, DataType::Double>();
                default: return CreateDecoder<EncodingType::Plain, DataType::Int32>();
            }
        case EncodingType::RLE:
            switch (dt) {
                case DataType::Int32: return CreateDecoder<EncodingType::RLE, DataType::Int32>();
                case DataType::Int64: return CreateDecoder<EncodingType::RLE, DataType::Int64>();
                case DataType::Float: return CreateDecoder<EncodingType::RLE, DataType::Float>();
                case DataType::Double: return CreateDecoder<EncodingType::RLE, DataType::Double>();
                default: return CreateDecoder<EncodingType::Plain, DataType::Int32>();
            }
        case EncodingType::Delta:
            switch (dt) {
                case DataType::Int32: return CreateDecoder<EncodingType::Delta, DataType::Int32>();
                case DataType::Int64: return CreateDecoder<EncodingType::Delta, DataType::Int64>();
                default: return CreateDecoder<EncodingType::Plain, DataType::Int32>();
            }
        case EncodingType::FOR:
            switch (dt) {
                case DataType::Int32: return CreateDecoder<EncodingType::FOR, DataType::Int32>();
                case DataType::Int64: return CreateDecoder<EncodingType::FOR, DataType::Int64>();
                default: return CreateDecoder<EncodingType::Plain, DataType::Int32>();
            }
        case EncodingType::Dictionary:
            switch (dt) {
                case DataType::Int32: return CreateDecoder<EncodingType::Dictionary, DataType::Int32>();
                case DataType::Int64: return CreateDecoder<EncodingType::Dictionary, DataType::Int64>();
                case DataType::Float: return CreateDecoder<EncodingType::Dictionary, DataType::Float>();
                case DataType::Double: return CreateDecoder<EncodingType::Dictionary, DataType::Double>();
                default: return CreateDecoder<EncodingType::Plain, DataType::Int32>();
            }
        case EncodingType::BitPack:
            switch (dt) {
                case DataType::Int32: return CreateDecoder<EncodingType::BitPack, DataType::Int32>();
                case DataType::Int64: return CreateDecoder<EncodingType::BitPack, DataType::Int64>();
                default: return CreateDecoder<EncodingType::Plain, DataType::Int32>();
            }
        default: return CreateDecoder<EncodingType::Plain, DataType::Int32>();
    }
}

std::unique_ptr<Encoder> CreateZstdEncoder(int level = 3);
std::unique_ptr<Decoder> CreateZstdDecoder();
std::unique_ptr<Encoder> CreateLz4Encoder(int level = 9);
std::unique_ptr<Decoder> CreateLz4Decoder();

} // namespace encoding
} // namespace columnar
