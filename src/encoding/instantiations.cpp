#include <columnar/encoding/encoding.h>

namespace columnar {
namespace encoding {

// Only instantiate what's actually defined in the header
// Plain: all types
template std::unique_ptr<Encoder> CreateEncoder<EncodingType::Plain, DataType::Int32>();
template std::unique_ptr<Encoder> CreateEncoder<EncodingType::Plain, DataType::Int64>();
template std::unique_ptr<Encoder> CreateEncoder<EncodingType::Plain, DataType::Float>();
template std::unique_ptr<Encoder> CreateEncoder<EncodingType::Plain, DataType::Double>();
template std::unique_ptr<Decoder> CreateDecoder<EncodingType::Plain, DataType::Int32>();
template std::unique_ptr<Decoder> CreateDecoder<EncodingType::Plain, DataType::Int64>();
template std::unique_ptr<Decoder> CreateDecoder<EncodingType::Plain, DataType::Float>();
template std::unique_ptr<Decoder> CreateDecoder<EncodingType::Plain, DataType::Double>();

// RLE: Int32, Int64
template std::unique_ptr<Encoder> CreateEncoder<EncodingType::RLE, DataType::Int32>();
template std::unique_ptr<Encoder> CreateEncoder<EncodingType::RLE, DataType::Int64>();
template std::unique_ptr<Decoder> CreateDecoder<EncodingType::RLE, DataType::Int32>();
template std::unique_ptr<Decoder> CreateDecoder<EncodingType::RLE, DataType::Int64>();

// Delta: Int32, Int64
template std::unique_ptr<Encoder> CreateEncoder<EncodingType::Delta, DataType::Int32>();
template std::unique_ptr<Encoder> CreateEncoder<EncodingType::Delta, DataType::Int64>();
template std::unique_ptr<Decoder> CreateDecoder<EncodingType::Delta, DataType::Int32>();
template std::unique_ptr<Decoder> CreateDecoder<EncodingType::Delta, DataType::Int64>();

// FOR: Int32, Int64
template std::unique_ptr<Encoder> CreateEncoder<EncodingType::FOR, DataType::Int32>();
template std::unique_ptr<Encoder> CreateEncoder<EncodingType::FOR, DataType::Int64>();
template std::unique_ptr<Decoder> CreateDecoder<EncodingType::FOR, DataType::Int32>();
template std::unique_ptr<Decoder> CreateDecoder<EncodingType::FOR, DataType::Int64>();

// Dictionary: Int32, Int64, Float, Double
template std::unique_ptr<Encoder> CreateEncoder<EncodingType::Dictionary, DataType::Int32>();
template std::unique_ptr<Encoder> CreateEncoder<EncodingType::Dictionary, DataType::Int64>();
template std::unique_ptr<Encoder> CreateEncoder<EncodingType::Dictionary, DataType::Float>();
template std::unique_ptr<Encoder> CreateEncoder<EncodingType::Dictionary, DataType::Double>();
template std::unique_ptr<Decoder> CreateDecoder<EncodingType::Dictionary, DataType::Int32>();
template std::unique_ptr<Decoder> CreateDecoder<EncodingType::Dictionary, DataType::Int64>();
template std::unique_ptr<Decoder> CreateDecoder<EncodingType::Dictionary, DataType::Float>();
template std::unique_ptr<Decoder> CreateDecoder<EncodingType::Dictionary, DataType::Double>();

// BitPack: Int32, Int64
template std::unique_ptr<Encoder> CreateEncoder<EncodingType::BitPack, DataType::Int32>();
template std::unique_ptr<Encoder> CreateEncoder<EncodingType::BitPack, DataType::Int64>();
template std::unique_ptr<Decoder> CreateDecoder<EncodingType::BitPack, DataType::Int32>();
template std::unique_ptr<Decoder> CreateDecoder<EncodingType::BitPack, DataType::Int64>();

} // namespace encoding
} // namespace columnar
