#include <columnar/for.h>
#include <immintrin.h>

namespace columnar {
namespace for_encoding {

template class ForEncoder<DataType::Int32>;
template class ForEncoder<DataType::Int64>;
template class ForDecoder<DataType::Int32>;
template class ForDecoder<DataType::Int64>;

} // namespace for_encoding
} // namespace columnar
