#include <columnar/rle.h>
#include <algorithm>
#include <immintrin.h>

namespace columnar {
namespace rle {

template class RleEncoder<DataType::Int32>;
template class RleEncoder<DataType::Int64>;
template class RleEncoder<DataType::Float>;
template class RleEncoder<DataType::Double>;
template class RleDecoder<DataType::Int32>;
template class RleDecoder<DataType::Int64>;
template class RleDecoder<DataType::Float>;
template class RleDecoder<DataType::Double>;

} // namespace rle
} // namespace columnar
