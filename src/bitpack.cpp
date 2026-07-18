#include <columnar/bitpack.h>
#include <immintrin.h>

namespace columnar {
namespace bitpack {

template class BitPackEncoder<DataType::Int32>;
template class BitPackEncoder<DataType::Int64>;
template class BitPackDecoder<DataType::Int32>;
template class BitPackDecoder<DataType::Int64>;

} // namespace bitpack
} // namespace columnar