#pragma once

#include <cstdint>
#include <cstddef>
#include <string_view>
#include <variant>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

namespace columnar {

enum class DataType : uint8_t {
    Boolean = 0,
    Int32 = 1,
    Int64 = 2,
    Float = 3,
    Double = 4,
    Varchar = 5
};

constexpr size_t TypeSize(DataType type) {
    switch (type) {
        case DataType::Boolean: return 1;
        case DataType::Int32: return 4;
        case DataType::Int64: return 8;
        case DataType::Float: return 4;
        case DataType::Double: return 8;
        case DataType::Varchar: return 16;
    }
    return 0;
}

constexpr bool IsFixedSize(DataType type) {
    return type != DataType::Varchar;
}

constexpr bool IsInteger(DataType type) {
    return type == DataType::Int32 || type == DataType::Int64;
}

constexpr bool IsFloatingPoint(DataType type) {
    return type == DataType::Float || type == DataType::Double;
}

constexpr bool IsNumeric(DataType type) {
    return IsInteger(type) || IsFloatingPoint(type);
}

constexpr bool IsSigned(DataType type) {
    return type == DataType::Int32 || type == DataType::Int64 ||
           type == DataType::Float || type == DataType::Double;
}

struct VarcharView {
    const char* data;
    uint32_t length;
    
    VarcharView() : data(nullptr), length(0) {}
    VarcharView(const char* d, uint32_t l) : data(d), length(l) {}
    VarcharView(const std::string_view& sv) : data(sv.data()), length(static_cast<uint32_t>(sv.size())) {}
    
    std::string_view view() const { return std::string_view(data, length); }
    bool operator==(const VarcharView& other) const {
        return length == other.length && std::memcmp(data, other.data, length) == 0;
    }
    bool operator<(const VarcharView& other) const {
        return view() < other.view();
    }
};

using Value = std::variant<
    bool,
    int32_t,
    int64_t,
    float,
    double,
    VarcharView
>;

template<DataType T>
struct TypeTraits;

template<>
struct TypeTraits<DataType::Boolean> { using Type = bool; static constexpr DataType type = DataType::Boolean; };

template<>
struct TypeTraits<DataType::Int32> { using Type = int32_t; static constexpr DataType type = DataType::Int32; };

template<>
struct TypeTraits<DataType::Int64> { using Type = int64_t; static constexpr DataType type = DataType::Int64; };

template<>
struct TypeTraits<DataType::Float> { using Type = float; static constexpr DataType type = DataType::Float; };

template<>
struct TypeTraits<DataType::Double> { using Type = double; static constexpr DataType type = DataType::Double; };

template<>
struct TypeTraits<DataType::Varchar> { using Type = VarcharView; static constexpr DataType type = DataType::Varchar; };

template<DataType T>
using NativeType = typename TypeTraits<T>::Type;

// ponytail: removed NativeType() constexpr function that conflicted with the using alias above

constexpr size_t kBlockSize = 8192;
constexpr size_t kSegmentRows = 65536;
constexpr size_t kDeltaFlushThreshold = 100000;
constexpr size_t kBloomFilterBits = 1 << 20;
constexpr size_t kBloomHashFunctions = 7;
constexpr size_t kZoneMapBlockRows = 65536;

enum class EncodingType : uint8_t {
    Plain = 0,
    Dictionary = 1,
    RLE = 2,
    BitPack = 3,
    FOR = 4,
    ZSTD = 5,
    LZ4 = 6,
    Delta = 7
};

enum class PredicateType : uint8_t {
    Equal = 0,
    NotEqual = 1,
    LessThan = 2,
    LessEqual = 3,
    GreaterThan = 4,
    GreaterEqual = 5,
    IsNull = 6,
    IsNotNull = 7,
    In = 8,
    NotIn = 9,
    And = 10,
    Or = 11,
    Not = 12
};

struct ColumnStats {
    DataType type = DataType::Int32;
    uint64_t null_count = 0;
    uint64_t distinct_count = 0;
    uint64_t total_count = 0;
    
    int64_t min_int = 0;
    int64_t max_int = 0;
    double min_fp = 0.0;
    double max_fp = 0.0;
    VarcharView min_str;
    VarcharView max_str;
    
    bool has_min_max = false;
    
ColumnStats() = default;
    ~ColumnStats() = default;

    ColumnStats(DataType t, uint64_t nc, uint64_t dc, uint64_t tc, bool hmm)
        : type(t), null_count(nc), distinct_count(dc), total_count(tc), has_min_max(hmm) {}

    ColumnStats(DataType t, uint64_t nc, uint64_t dc, uint64_t tc, int64_t min_v, int64_t max_v)
        : type(t), null_count(nc), distinct_count(dc), total_count(tc), min_int(min_v), max_int(max_v), has_min_max(true) {}

    ColumnStats(const ColumnStats& other) {
        type = other.type;
        null_count = other.null_count;
        distinct_count = other.distinct_count;
        total_count = other.total_count;
        has_min_max = other.has_min_max;
        if (has_min_max && IsInteger(type)) {
            min_int = other.min_int;
            max_int = other.max_int;
        } else if (has_min_max && IsFloatingPoint(type)) {
            min_fp = other.min_fp;
            max_fp = other.max_fp;
        }
    }
    
    ColumnStats& operator=(const ColumnStats& other) {
        if (this != &other) {
            type = other.type;
            null_count = other.null_count;
            distinct_count = other.distinct_count;
            total_count = other.total_count;
            has_min_max = other.has_min_max;
            if (has_min_max && IsInteger(type)) {
                min_int = other.min_int;
                max_int = other.max_int;
            } else if (has_min_max && IsFloatingPoint(type)) {
                min_fp = other.min_fp;
                max_fp = other.max_fp;
            }
        }
        return *this;
    }
    
    template<DataType T>
    void UpdateMinMax(const NativeType<T>& value) {
        if (!has_min_max) {
            has_min_max = true;
            if constexpr (IsInteger(T)) min_int = static_cast<int64_t>(value);
            else if constexpr (IsFloatingPoint(T)) min_fp = static_cast<double>(value);
        } else {
            if constexpr (IsInteger(T)) {
                min_int = std::min(min_int, static_cast<int64_t>(value));
                max_int = std::max(max_int, static_cast<int64_t>(value));
            } else if constexpr (IsFloatingPoint(T)) {
                min_fp = std::min(min_fp, static_cast<double>(value));
                max_fp = std::max(max_fp, static_cast<double>(value));
            }
        }
    }
};

// SegmentHeader is defined in segment.h

struct ZoneMapEntry {
    uint64_t min_int;
    uint64_t max_int;
    uint32_t null_count;
    uint32_t row_count;
    bool has_nulls;
    
    ZoneMapEntry() : min_int(0), max_int(0), null_count(0), row_count(0), has_nulls(false) {}
    
    template<DataType T>
    bool CanMatch(PredicateType pred, const NativeType<T>& value) const {
        if (row_count == 0) return false;
        if (pred == PredicateType::IsNull) return has_nulls;
        if (pred == PredicateType::IsNotNull) return row_count > null_count;
        
        if constexpr (IsInteger(T) || IsFloatingPoint(T)) {
            int64_t min_v = static_cast<int64_t>(min_int);
            int64_t max_v = static_cast<int64_t>(max_int);
            int64_t val = static_cast<int64_t>(value);
            
            switch (pred) {
                case PredicateType::Equal: return min_v <= val && val <= max_v;
                case PredicateType::NotEqual: return !(min_v == val && max_v == val);
                case PredicateType::LessThan: return min_v < val;
                case PredicateType::LessEqual: return min_v <= val;
                case PredicateType::GreaterThan: return max_v > val;
                case PredicateType::GreaterEqual: return max_v >= val;
                default: return true;
            }
        }
        return true;
    }
};

template<typename T>
concept Integral = std::is_integral_v<T>;

template<typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

template<typename T>
concept Arithmetic = Integral<T> || FloatingPoint<T>;

} // namespace columnar
