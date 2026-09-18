#pragma once

#include <columnar/types.h>
#include <columnar/encoding/encoding.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

namespace columnar {
namespace zone_map {

struct ZoneMapEntry {
    int64_t min_int;
    int64_t max_int;
    double min_fp;
    double max_fp;
    uint32_t null_count;
    uint32_t row_count;
    bool has_min_max;
    bool has_nulls;
    
    ZoneMapEntry() : min_int(0), max_int(0), min_fp(0), max_fp(0), 
                     null_count(0), row_count(0), has_min_max(false), has_nulls(false) {}
    
    template<DataType T>
    void Update(const typename TypeTraits<T>::Type* data, size_t count, const bool* nulls) {
        if (count == 0) return;
        
        if constexpr (IsInteger(T) || IsFloatingPoint(T)) {
            using ValueType = typename TypeTraits<T>::Type;
            
            size_t first_valid = 0;
            if (nulls) {
                while (first_valid < count && nulls[first_valid]) ++first_valid;
            }
            
            if (first_valid >= count) {
                null_count = static_cast<uint32_t>(count);
                row_count = static_cast<uint32_t>(count);
                has_nulls = true;
                return;
            }
            
            ValueType min_val = data[first_valid];
            ValueType max_val = data[first_valid];
            
            for (size_t i = first_valid + 1; i < count; ++i) {
                if (nulls && nulls[i]) {
                    ++null_count;
                    continue;
                }
                if (data[i] < min_val) min_val = data[i];
                if (data[i] > max_val) max_val = data[i];
            }
            
            if constexpr (IsInteger(T)) {
                min_int = static_cast<int64_t>(min_val);
                max_int = static_cast<int64_t>(max_val);
            } else {
                min_fp = static_cast<double>(min_val);
                max_fp = static_cast<double>(max_val);
            }
            
            row_count = static_cast<uint32_t>(count);
            has_min_max = true;
            has_nulls = null_count > 0;
        }
    }
    
    template<DataType T, PredicateType Pred>
    bool CanMatch(const typename TypeTraits<T>::Type& value) const {
        if (!has_min_max && row_count > 0) return true;
        
        if (Pred == PredicateType::IsNull) return has_nulls;
        if (Pred == PredicateType::IsNotNull) return row_count > null_count;
        if (row_count == 0) return false;
        
        if constexpr (IsInteger(T)) {
            int64_t val = static_cast<int64_t>(value);
            switch (Pred) {
                case PredicateType::Equal: return min_int <= val && val <= max_int;
                case PredicateType::NotEqual: return !(min_int == val && max_int == val);
                case PredicateType::LessThan: return min_int < val;
                case PredicateType::LessEqual: return min_int <= val;
                case PredicateType::GreaterThan: return max_int > val;
                case PredicateType::GreaterEqual: return max_int >= val;
                default: return true;
            }
        } else if constexpr (IsFloatingPoint(T)) {
            double val = static_cast<double>(value);
            switch (Pred) {
                case PredicateType::Equal: return min_fp <= val && val <= max_fp;
                case PredicateType::NotEqual: return !(min_fp == val && max_fp == val);
                case PredicateType::LessThan: return min_fp < val;
                case PredicateType::LessEqual: return min_fp <= val;
                case PredicateType::GreaterThan: return max_fp > val;
                case PredicateType::GreaterEqual: return max_fp >= val;
                default: return true;
            }
        }
        return true;
    }
    
    bool CanMatch(DataType type, PredicateType pred, const void* value) const {
        switch (type) {
            case DataType::Int32: 
                switch (pred) {
                    case PredicateType::Equal: return CanMatch<DataType::Int32, PredicateType::Equal>(*static_cast<const int32_t*>(value));
                    case PredicateType::NotEqual: return CanMatch<DataType::Int32, PredicateType::NotEqual>(*static_cast<const int32_t*>(value));
                    case PredicateType::LessThan: return CanMatch<DataType::Int32, PredicateType::LessThan>(*static_cast<const int32_t*>(value));
                    case PredicateType::LessEqual: return CanMatch<DataType::Int32, PredicateType::LessEqual>(*static_cast<const int32_t*>(value));
                    case PredicateType::GreaterThan: return CanMatch<DataType::Int32, PredicateType::GreaterThan>(*static_cast<const int32_t*>(value));
                    case PredicateType::GreaterEqual: return CanMatch<DataType::Int32, PredicateType::GreaterEqual>(*static_cast<const int32_t*>(value));
                    case PredicateType::IsNull: return has_nulls;
                    case PredicateType::IsNotNull: return row_count > null_count;
                    default: return true;
                }
            case DataType::Int64: 
                switch (pred) {
                    case PredicateType::Equal: return CanMatch<DataType::Int64, PredicateType::Equal>(*static_cast<const int64_t*>(value));
                    case PredicateType::NotEqual: return CanMatch<DataType::Int64, PredicateType::NotEqual>(*static_cast<const int64_t*>(value));
                    case PredicateType::LessThan: return CanMatch<DataType::Int64, PredicateType::LessThan>(*static_cast<const int64_t*>(value));
                    case PredicateType::LessEqual: return CanMatch<DataType::Int64, PredicateType::LessEqual>(*static_cast<const int64_t*>(value));
                    case PredicateType::GreaterThan: return CanMatch<DataType::Int64, PredicateType::GreaterThan>(*static_cast<const int64_t*>(value));
                    case PredicateType::GreaterEqual: return CanMatch<DataType::Int64, PredicateType::GreaterEqual>(*static_cast<const int64_t*>(value));
                    case PredicateType::IsNull: return has_nulls;
                    case PredicateType::IsNotNull: return row_count > null_count;
                    default: return true;
                }
            case DataType::Float: 
                switch (pred) {
                    case PredicateType::Equal: return CanMatch<DataType::Float, PredicateType::Equal>(*static_cast<const float*>(value));
                    case PredicateType::NotEqual: return CanMatch<DataType::Float, PredicateType::NotEqual>(*static_cast<const float*>(value));
                    case PredicateType::LessThan: return CanMatch<DataType::Float, PredicateType::LessThan>(*static_cast<const float*>(value));
                    case PredicateType::LessEqual: return CanMatch<DataType::Float, PredicateType::LessEqual>(*static_cast<const float*>(value));
                    case PredicateType::GreaterThan: return CanMatch<DataType::Float, PredicateType::GreaterThan>(*static_cast<const float*>(value));
                    case PredicateType::GreaterEqual: return CanMatch<DataType::Float, PredicateType::GreaterEqual>(*static_cast<const float*>(value));
                    case PredicateType::IsNull: return has_nulls;
                    case PredicateType::IsNotNull: return row_count > null_count;
                    default: return true;
                }
            case DataType::Double: 
                switch (pred) {
                    case PredicateType::Equal: return CanMatch<DataType::Double, PredicateType::Equal>(*static_cast<const double*>(value));
                    case PredicateType::NotEqual: return CanMatch<DataType::Double, PredicateType::NotEqual>(*static_cast<const double*>(value));
                    case PredicateType::LessThan: return CanMatch<DataType::Double, PredicateType::LessThan>(*static_cast<const double*>(value));
                    case PredicateType::LessEqual: return CanMatch<DataType::Double, PredicateType::LessEqual>(*static_cast<const double*>(value));
                    case PredicateType::GreaterThan: return CanMatch<DataType::Double, PredicateType::GreaterThan>(*static_cast<const double*>(value));
                    case PredicateType::GreaterEqual: return CanMatch<DataType::Double, PredicateType::GreaterEqual>(*static_cast<const double*>(value));
                    case PredicateType::IsNull: return has_nulls;
                    case PredicateType::IsNotNull: return row_count > null_count;
                    default: return true;
                }
            default: return true;
        }
    }
};

class ZoneMap {
    std::vector<ZoneMapEntry> entries_;
    size_t block_rows_;
    
public:
    explicit ZoneMap(size_t block_rows = 65536) : block_rows_(block_rows) {}
    
    void AddBlock(const ZoneMapEntry& entry) {
        entries_.push_back(entry);
    }
    
    template<DataType T>
    void BuildFromColumn(const typename TypeTraits<T>::Type* data, const bool* nulls, size_t count) {
        size_t num_blocks = (count + block_rows_ - 1) / block_rows_;
        entries_.resize(num_blocks);
        
        for (size_t b = 0; b < num_blocks; ++b) {
            size_t start = b * block_rows_;
            size_t end = std::min(start + block_rows_, count);
            entries_[b].template Update<T>(data + start, end - start, nulls ? nulls + start : nullptr);
        }
    }
    
    size_t NumBlocks() const { return entries_.size(); }
    const ZoneMapEntry& Block(size_t idx) const { return entries_[idx]; }
    ZoneMapEntry& Block(size_t idx) { return entries_[idx]; }
    
    template<PredicateType Pred, DataType T>
    bool ShouldScanBlock(size_t block_idx, const typename TypeTraits<T>::Type& value) const {
        return entries_[block_idx].template CanMatch<T, Pred>(value);
    }
    
    bool ShouldScanBlock(size_t block_idx, DataType type, PredicateType pred, const void* value) const {
        if (block_idx >= entries_.size()) return false;
        
        switch (type) {
            case DataType::Int32:
                switch (pred) {
                    case PredicateType::Equal: return ShouldScanBlock<PredicateType::Equal, DataType::Int32>(block_idx, *static_cast<const int32_t*>(value));
                    case PredicateType::NotEqual: return ShouldScanBlock<PredicateType::NotEqual, DataType::Int32>(block_idx, *static_cast<const int32_t*>(value));
                    case PredicateType::LessThan: return ShouldScanBlock<PredicateType::LessThan, DataType::Int32>(block_idx, *static_cast<const int32_t*>(value));
                    case PredicateType::LessEqual: return ShouldScanBlock<PredicateType::LessEqual, DataType::Int32>(block_idx, *static_cast<const int32_t*>(value));
                    case PredicateType::GreaterThan: return ShouldScanBlock<PredicateType::GreaterThan, DataType::Int32>(block_idx, *static_cast<const int32_t*>(value));
                    case PredicateType::GreaterEqual: return ShouldScanBlock<PredicateType::GreaterEqual, DataType::Int32>(block_idx, *static_cast<const int32_t*>(value));
                    case PredicateType::IsNull: return entries_[block_idx].has_nulls;
                    case PredicateType::IsNotNull: return entries_[block_idx].row_count > entries_[block_idx].null_count;
                    default: return true;
                }
            case DataType::Int64:
                switch (pred) {
                    case PredicateType::Equal: return ShouldScanBlock<PredicateType::Equal, DataType::Int64>(block_idx, *static_cast<const int64_t*>(value));
                    case PredicateType::NotEqual: return ShouldScanBlock<PredicateType::NotEqual, DataType::Int64>(block_idx, *static_cast<const int64_t*>(value));
                    case PredicateType::LessThan: return ShouldScanBlock<PredicateType::LessThan, DataType::Int64>(block_idx, *static_cast<const int64_t*>(value));
                    case PredicateType::LessEqual: return ShouldScanBlock<PredicateType::LessEqual, DataType::Int64>(block_idx, *static_cast<const int64_t*>(value));
                    case PredicateType::GreaterThan: return ShouldScanBlock<PredicateType::GreaterThan, DataType::Int64>(block_idx, *static_cast<const int64_t*>(value));
                    case PredicateType::GreaterEqual: return ShouldScanBlock<PredicateType::GreaterEqual, DataType::Int64>(block_idx, *static_cast<const int64_t*>(value));
                    case PredicateType::IsNull: return entries_[block_idx].has_nulls;
                    case PredicateType::IsNotNull: return entries_[block_idx].row_count > entries_[block_idx].null_count;
                    default: return true;
                }
            case DataType::Float:
                switch (pred) {
                    case PredicateType::Equal: return ShouldScanBlock<PredicateType::Equal, DataType::Float>(block_idx, *static_cast<const float*>(value));
                    case PredicateType::NotEqual: return ShouldScanBlock<PredicateType::NotEqual, DataType::Float>(block_idx, *static_cast<const float*>(value));
                    case PredicateType::LessThan: return ShouldScanBlock<PredicateType::LessThan, DataType::Float>(block_idx, *static_cast<const float*>(value));
                    case PredicateType::LessEqual: return ShouldScanBlock<PredicateType::LessEqual, DataType::Float>(block_idx, *static_cast<const float*>(value));
                    case PredicateType::GreaterThan: return ShouldScanBlock<PredicateType::GreaterThan, DataType::Float>(block_idx, *static_cast<const float*>(value));
                    case PredicateType::GreaterEqual: return ShouldScanBlock<PredicateType::GreaterEqual, DataType::Float>(block_idx, *static_cast<const float*>(value));
                    case PredicateType::IsNull: return entries_[block_idx].has_nulls;
                    case PredicateType::IsNotNull: return entries_[block_idx].row_count > entries_[block_idx].null_count;
                    default: return true;
                }
            case DataType::Double:
                switch (pred) {
                    case PredicateType::Equal: return ShouldScanBlock<PredicateType::Equal, DataType::Double>(block_idx, *static_cast<const double*>(value));
                    case PredicateType::NotEqual: return ShouldScanBlock<PredicateType::NotEqual, DataType::Double>(block_idx, *static_cast<const double*>(value));
                    case PredicateType::LessThan: return ShouldScanBlock<PredicateType::LessThan, DataType::Double>(block_idx, *static_cast<const double*>(value));
                    case PredicateType::LessEqual: return ShouldScanBlock<PredicateType::LessEqual, DataType::Double>(block_idx, *static_cast<const double*>(value));
                    case PredicateType::GreaterThan: return ShouldScanBlock<PredicateType::GreaterThan, DataType::Double>(block_idx, *static_cast<const double*>(value));
                    case PredicateType::GreaterEqual: return ShouldScanBlock<PredicateType::GreaterEqual, DataType::Double>(block_idx, *static_cast<const double*>(value));
                    case PredicateType::IsNull: return entries_[block_idx].has_nulls;
                    case PredicateType::IsNotNull: return entries_[block_idx].row_count > entries_[block_idx].null_count;
                    default: return true;
                }
            default:
                return true;
        }
    }
    
    void Write(encoding::Buffer& out) const {
        out.Append(static_cast<uint32_t>(entries_.size()));
        out.Append(static_cast<uint32_t>(block_rows_));
        for (const auto& e : entries_) {
            out.Append(e.min_int);
            out.Append(e.max_int);
            out.Append(e.min_fp);
            out.Append(e.max_fp);
            out.Append(e.null_count);
            out.Append(e.row_count);
            out.Append(e.has_min_max);
            out.Append(e.has_nulls);
        }
    }
    
    static ZoneMap Read(const uint8_t* data, size_t /*size*/) {
        ZoneMap zm;
        const uint8_t* ptr = data;
        
        uint32_t num_blocks, block_rows;
        std::memcpy(&num_blocks, ptr, 4); ptr += 4;
        std::memcpy(&block_rows, ptr, 4); ptr += 4;
        
        zm.block_rows_ = block_rows;
        zm.entries_.resize(num_blocks);
        
        for (uint32_t i = 0; i < num_blocks; ++i) {
            ZoneMapEntry& e = zm.entries_[i];
            std::memcpy(&e.min_int, ptr, 8); ptr += 8;
            std::memcpy(&e.max_int, ptr, 8); ptr += 8;
            std::memcpy(&e.min_fp, ptr, 8); ptr += 8;
            std::memcpy(&e.max_fp, ptr, 8); ptr += 8;
            std::memcpy(&e.null_count, ptr, 4); ptr += 4;
            std::memcpy(&e.row_count, ptr, 4); ptr += 4;
            std::memcpy(&e.has_min_max, ptr, 1); ptr += 1;
            std::memcpy(&e.has_nulls, ptr, 1); ptr += 1;
        }
        
        return zm;
    }
};

} // namespace zone_map
} // namespace columnar
