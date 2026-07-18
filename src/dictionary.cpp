#include <columnar/dictionary.h>
#include <algorithm>
#include <unordered_map>

namespace columnar {
namespace dictionary {

template class DictionaryEncoder<DataType::Int32>;
template class DictionaryEncoder<DataType::Int64>;
template class DictionaryEncoder<DataType::Float>;
template class DictionaryEncoder<DataType::Double>;
template class DictionaryDecoder<DataType::Int32>;
template class DictionaryDecoder<DataType::Int64>;
template class DictionaryDecoder<DataType::Float>;
template class DictionaryDecoder<DataType::Double>;

} // namespace dictionary
} // namespace columnar