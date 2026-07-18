#pragma once

#include <columnar/types.h>
#include <columnar/encoding/encoding.h>
#include <memory>

namespace columnar {
namespace encoding {

template<EncodingType Type, DataType T>
std::unique_ptr<Encoder> CreateEncoder();

template<>
inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Plain, DataType::Int32>() {
    return std::make_unique<PlainEncoder<DataType::Int32>>();
}

template<>
inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Plain, DataType::Int64>() {
    return std::make_unique<PlainEncoder<DataType::Int64>>();
}

template<>
inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Plain, DataType::Float>() {
    return std::make_unique<PlainEncoder<DataType::Float>>();
}

template<>
inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Plain, DataType::Double>() {
    return std::make_unique<PlainEncoder<DataType::Double>>();
}

template<>
inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::RLE, DataType::Int32>() {
    return std::make_unique<RLEEncoder<DataType::Int32>>();
}

template<>
inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::RLE, DataType::Int64>() {
    return std::make_unique<RLEEncoder<DataType::Int64>>();
}

template<>
inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Delta, DataType::Int32>() {
    return std::make_unique<DeltaEncoder<DataType::Int32>>();
}

template<>
inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Delta, DataType::Int64>() {
    return std::make_unique<DeltaEncoder<DataType::Int64>>();
}

template<>
inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::FOR, DataType::Int32>() {
    return std::make_unique<FOREncoder<DataType::Int32>>();
}

template<>
inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::FOR, DataType::Int64>() {
    return std::make_unique<FOREncoder<DataType::Int64>>();
}

template<>
inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Dictionary, DataType::Int32>() {
    return std::make_unique<DictEncoder<DataType::Int32>>();
}

template<>
inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::Dictionary, DataType::Int64>() {
    return std::make_unique<DictEncoder<DataType::Int64>>();
}

template<>
inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::BitPack, DataType::Int32>() {
    return std::make_unique<BitPackEncoder<DataType::Int32, 32>>();
}

template<>
inline std::unique_ptr<Encoder> CreateEncoder<EncodingType::BitPack, DataType::Int64>() {
    return std::make_unique<BitPackEncoder<DataType::Int64, 64>>();
}

template<EncodingType Type, DataType T>
std::unique_ptr<Decoder> CreateDecoder();

template<>
inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Plain, DataType::Int32>() {
    return std::make_unique<PlainDecoder<DataType::Int32>>();
}

template<>
inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Plain, DataType::Int64>() {
    return std::make_unique<PlainDecoder<DataType::Int64>>();
}

template<>
inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Plain, DataType::Float>() {
    return std::make_unique<PlainDecoder<DataType::Float>>();
}

template<>
inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Plain, DataType::Double>() {
    return std::make_unique<PlainDecoder<DataType::Double>>();
}

template<>
inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::RLE, DataType::Int32>() {
    return std::make_unique<RLEDecoder<DataType::Int32>>();
}

template<>
inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::RLE, DataType::Int64>() {
    return std::make_unique<RLEDecoder<DataType::Int64>>();
}

template<>
inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Delta, DataType::Int32>() {
    return std::make_unique<DeltaDecoder<DataType::Int32>>();
}

template<>
inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Delta, DataType::Int64>() {
    return std::make_unique<DeltaDecoder<DataType::Int64>>();
}

template<>
inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::FOR, DataType::Int32>() {
    return std::make_unique<FORDecoder<DataType::Int32>>();
}

template<>
inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::FOR, DataType::Int64>() {
    return std::make_unique<FORDecoder<DataType::Int64>>();
}

template<>
inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Dictionary, DataType::Int32>() {
    return std::make_unique<DictDecoder<DataType::Int32>>();
}

template<>
inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::Dictionary, DataType::Int64>() {
    return std::make_unique<DictDecoder<DataType::Int64>>();
}

template<>
inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::BitPack, DataType::Int32>() {
    return std::make_unique<BitPackDecoder<DataType::Int32, 32>>();
}

template<>
inline std::unique_ptr<Decoder> CreateDecoder<EncodingType::BitPack, DataType::Int64>() {
    return std::make_unique<BitPackDecoder<DataType::Int64, 64>>();
}

inline std::unique_ptr<Encoder> CreateEncoder(EncodingType type, DataType data_type) {
    switch (type) {
        case EncodingType::Plain:
            switch (data_type) {
                case DataType::Int32: return CreateEncoder<EncodingType::Plain, DataType::Int32>();
                case DataType::Int64: return CreateEncoder<EncodingType::Plain, DataType::Int64>();
                case DataType::Float: return CreateEncoder<EncodingType::Plain, DataType::Float>();
                case DataType::Double: return CreateEncoder<EncodingType::Plain, DataType::Double>();
                default: return nullptr;
            }
        case EncodingType::RLE:
            switch (data_type) {
                case DataType::Int32: return CreateEncoder<EncodingType::RLE, DataType::Int32>();
                case DataType::Int64: return CreateEncoder<EncodingType::RLE, DataType::Int64>();
                default: return nullptr;
            }
        case EncodingType::Delta:
            switch (data_type) {
                case DataType::Int32: return CreateEncoder<EncodingType::Delta, DataType::Int32>();
                case DataType::Int64: return CreateEncoder<EncodingType::Delta, DataType::Int64>();
                default: return nullptr;
            }
        case EncodingType::FOR:
            switch (data_type) {
                case DataType::Int32: return CreateEncoder<EncodingType::FOR, DataType::Int32>();
                case DataType::Int64: return CreateEncoder<EncodingType::FOR, DataType::Int64>();
                default: return nullptr;
            }
        case EncodingType::Dictionary:
            switch (data_type) {
                case DataType::Int32: return CreateEncoder<EncodingType::Dictionary, DataType::Int32>();
                case DataType::Int64: return CreateEncoder<EncodingType::Dictionary, DataType::Int64>();
                default: return nullptr;
            }
        case EncodingType::BitPack:
            switch (data_type) {
                case DataType::Int32: return CreateEncoder<EncodingType::BitPack, DataType::Int32>();
                case DataType::Int64: return CreateEncoder<EncodingType::BitPack, DataType::Int64>();
                default: return nullptr;
            }
        default: return nullptr;
    }
}

inline std::unique_ptr<Decoder> CreateDecoder(EncodingType type, DataType data_type) {
    switch (type) {
        case EncodingType::Plain:
            switch (data_type) {
                case DataType::Int32: return CreateDecoder<EncodingType::Plain, DataType::Int32>();
                case DataType::Int64: return CreateDecoder<EncodingType::Plain, DataType::Int64>();
                case DataType::Float: return CreateDecoder<EncodingType::Plain, DataType::Float>();
                case DataType::Double: return CreateDecoder<EncodingType::Plain, DataType::Double>();
                default: return nullptr;
            }
        case EncodingType::RLE:
            switch (data_type) {
                case DataType::Int32: return CreateDecoder<EncodingType::RLE, DataType::Int32>();
                case DataType::Int64: return CreateDecoder<EncodingType::RLE, DataType::Int64>();
                default: return nullptr;
            }
        case EncodingType::Delta:
            switch (data_type) {
                case DataType::Int32: return CreateDecoder<EncodingType::Delta, DataType::Int32>();
                case DataType::Int64: return CreateDecoder<EncodingType::Delta, DataType::Int64>();
                default: return nullptr;
            }
        case EncodingType::FOR:
            switch (data_type) {
                case DataType::Int32: return CreateDecoder<EncodingType::FOR, DataType::Int32>();
                case DataType::Int64: return CreateDecoder<EncodingType::FOR, DataType::Int64>();
                default: return nullptr;
            }
        case EncodingType::Dictionary:
            switch (data_type) {
                case DataType::Int32: return CreateDecoder<EncodingType::Dictionary, DataType::Int32>();
                case DataType::Int64: return CreateDecoder<EncodingType::Dictionary, DataType::Int64>();
                default: return nullptr;
            }
        case EncodingType::BitPack:
            switch (data_type) {
                case DataType::Int32: return CreateDecoder<EncodingType::BitPack, DataType::Int32>();
                case DataType::Int64: return CreateDecoder<EncodingType::BitPack, DataType::Int64>();
                default: return nullptr;
            }
        default: return nullptr;
    }
}

} // namespace encoding
} // namespace columnar
