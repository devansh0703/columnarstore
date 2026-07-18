#include <columnar/encoding/encoding.h>
#include <zstd.h>
#include <lz4.h>
#include <lz4hc.h>
#include <vector>
#include <cstring>

namespace columnar {
namespace encoding {

class ZstdEncoder : public Encoder {
    int compression_level_ = 3;
    
public:
    ZstdEncoder(int level = 3) : compression_level_(level) {}
    EncodingType Type() const override { return EncodingType::ZSTD; }
    
    void Encode(const void* data, size_t count, Buffer& out) override {
        size_t src_size = count;
        size_t bound = ZSTD_compressBound(src_size);
        std::vector<uint8_t> tmp(bound);
        
        size_t compressed = ZSTD_compress(tmp.data(), bound, data, src_size, compression_level_);
        if (ZSTD_isError(compressed)) return;
        out.Append(tmp.data(), compressed);
    }
};

class ZstdDecoder : public Decoder {
public:
    EncodingType Type() const override { return EncodingType::ZSTD; }
    
    void Decode(const void* data, size_t size, void* out, size_t count) override {
        ZSTD_decompress(out, count, data, size);
    }
};

class Lz4Encoder : public Encoder {
    int compression_level_ = 9;
    
public:
    Lz4Encoder(int level = 9) : compression_level_(level) {}
    EncodingType Type() const override { return EncodingType::LZ4; }
    
    void Encode(const void* data, size_t count, Buffer& out) override {
        size_t src_size = count;
        size_t bound = LZ4_compressBound(static_cast<int>(src_size));
        std::vector<char> tmp(bound);
        
        int compressed = LZ4_compress_HC(static_cast<const char*>(data), 
                                          tmp.data(),
                                          static_cast<int>(src_size),
                                          static_cast<int>(bound),
                                          compression_level_);
        if (compressed <= 0) return;
        out.Append(tmp.data(), compressed);
    }
};

class Lz4Decoder : public Decoder {
public:
    EncodingType Type() const override { return EncodingType::LZ4; }
    
    void Decode(const void* data, size_t size, void* out, size_t count) override {
        LZ4_decompress_safe(static_cast<const char*>(data), static_cast<char*>(out), 
                           static_cast<int>(size), static_cast<int>(count));
    }
};

std::unique_ptr<Encoder> CreateZstdEncoder(int level) {
    return std::make_unique<ZstdEncoder>(level);
}

std::unique_ptr<Decoder> CreateZstdDecoder() {
    return std::make_unique<ZstdDecoder>();
}

std::unique_ptr<Encoder> CreateLz4Encoder(int level) {
    return std::make_unique<Lz4Encoder>(level);
}

std::unique_ptr<Decoder> CreateLz4Decoder() {
    return std::make_unique<Lz4Decoder>();
}

} // namespace encoding
} // namespace columnar
