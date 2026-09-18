#pragma once

#include <columnar/types.h>
#include <columnar/segment.h>
#include <columnar/predicate.h>
#include <columnar/zone_map.h>
#include <columnar/bloom_filter.h>
#include <columnar/encoding/encoding.h>
#include <columnar/column/column.h>
#include <columnar/delta_store.h>
#include <vector>
#include <memory>
#include <cstdint>
#include <functional>
#include <atomic>
#include <chrono>
#include <thread>
#include <immintrin.h>

namespace columnar {
namespace scanner {

constexpr size_t kScanBlockSize = 8192;
constexpr size_t kVectorWidth = 16;

struct ScanContext {
    size_t block_size = kScanBlockSize;
    size_t num_threads = 1;
    std::vector<size_t> projected_columns;
    std::function<bool(size_t, const void*)> filter_callback;
    std::function<void(size_t, const void**, size_t)> project_callback;
    std::function<void(const void*, size_t)> aggregate_callback;
    bool enable_vectorization = true;
    bool enable_zone_map_pruning = true;
    bool enable_bloom_filter = true;
    bool late_materialization = true;
};

struct ScanStats {
    size_t blocks_scanned = 0;
    size_t blocks_pruned_zone_map = 0;
    size_t blocks_pruned_bloom = 0;
    size_t rows_scanned = 0;
    size_t rows_filtered = 0;
    size_t rows_returned = 0;
    double scan_time_ms = 0;
    double decode_time_ms = 0;
    double filter_time_ms = 0;
};

class BlockFilter {
    alignas(64) bool mask_[kScanBlockSize];
    
public:
    BlockFilter() { std::fill_n(mask_, kScanBlockSize, true); }
    
    bool* Data() { return mask_; }
    const bool* Data() const { return mask_; }
    
    void Reset() { std::fill_n(mask_, kScanBlockSize, true); }
    void SetAll(bool value) { std::fill_n(mask_, kScanBlockSize, value); }
    
    size_t CountTrue() const {
        size_t count = 0;
        for (size_t i = 0; i < kScanBlockSize; ++i) if (mask_[i]) ++count;
        return count;
    }
    
    void And(const BlockFilter& other) {
        for (size_t i = 0; i < kScanBlockSize; ++i) mask_[i] &= other.mask_[i];
    }
    
    void Or(const BlockFilter& other) {
        for (size_t i = 0; i < kScanBlockSize; ++i) mask_[i] |= other.mask_[i];
    }
    
    void Not() {
        for (size_t i = 0; i < kScanBlockSize; ++i) mask_[i] = !mask_[i];
    }
};

namespace detail {

// Widen a materialized column value to int64_t for bytecode evaluation.
inline int64_t WidenValue(const void* data, DataType type, size_t index) {
    switch (type) {
        case DataType::Int32: return static_cast<int64_t>(static_cast<const int32_t*>(data)[index]);
        case DataType::Int64: return static_cast<const int64_t*>(data)[index];
        case DataType::Float: return static_cast<int64_t>(static_cast<const float*>(data)[index]);
        case DataType::Double: return static_cast<int64_t>(static_cast<const double*>(data)[index]);
        default: return 0;
    }
}

}  // namespace detail

template<DataType T>
class VectorFilter {
    using ValueType = typename TypeTraits<T>::Type;

public:
#ifdef __AVX512F__
    static void Equal(const ValueType* data, const ValueType* value, bool* mask, size_t count) {
        if constexpr (sizeof(ValueType) == 4) {
            __m512i vval = _mm512_set1_epi32(*value);
            for (size_t i = 0; i + 15 < count; i += 16) {
                __m512i vdata = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data + i));
                __mmask16 m = _mm512_cmpeq_epi32_mask(vdata, vval);
                for (size_t j = 0; j < 16; ++j) mask[i + j] = (m >> j) & 1;
            }
        } else if constexpr (sizeof(ValueType) == 8) {
            __m512i vval = _mm512_set1_epi64(*value);
            for (size_t i = 0; i + 7 < count; i += 8) {
                __m512i vdata = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data + i));
                __mmask8 m = _mm512_cmpeq_epi64_mask(vdata, vval);
                for (size_t j = 0; j < 8; ++j) mask[i + j] = (m >> j) & 1;
            }
        }
        for (size_t i = count & ~15; i < count; ++i) mask[i] = (data[i] == *value);
    }

    static void LessThan(const ValueType* data, const ValueType* value, bool* mask, size_t count) {
        if constexpr (sizeof(ValueType) == 4) {
            __m512i vval = _mm512_set1_epi32(*value);
            for (size_t i = 0; i + 15 < count; i += 16) {
                __m512i vdata = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data + i));
                __mmask16 m = _mm512_cmplt_epi32_mask(vdata, vval);
                for (size_t j = 0; j < 16; ++j) mask[i + j] = (m >> j) & 1;
            }
        } else if constexpr (sizeof(ValueType) == 8) {
            __m512i vval = _mm512_set1_epi64(*value);
            for (size_t i = 0; i + 7 < count; i += 8) {
                __m512i vdata = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data + i));
                __mmask8 m = _mm512_cmplt_epi64_mask(vdata, vval);
                for (size_t j = 0; j < 8; ++j) mask[i + j] = (m >> j) & 1;
            }
        }
        for (size_t i = count & ~15; i < count; ++i) mask[i] = (data[i] < *value);
    }

    static void LessEqual(const ValueType* data, const ValueType* value, bool* mask, size_t count) {
        if constexpr (sizeof(ValueType) == 4) {
            __m512i vval = _mm512_set1_epi32(*value);
            for (size_t i = 0; i + 15 < count; i += 16) {
                __m512i vdata = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data + i));
                __mmask16 m = _mm512_cmple_epi32_mask(vdata, vval);
                for (size_t j = 0; j < 16; ++j) mask[i + j] = (m >> j) & 1;
            }
        } else if constexpr (sizeof(ValueType) == 8) {
            __m512i vval = _mm512_set1_epi64(*value);
            for (size_t i = 0; i + 7 < count; i += 8) {
                __m512i vdata = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data + i));
                __mmask8 m = _mm512_cmple_epi64_mask(vdata, vval);
                for (size_t j = 0; j < 8; ++j) mask[i + j] = (m >> j) & 1;
            }
        }
        for (size_t i = count & ~15; i < count; ++i) mask[i] = (data[i] <= *value);
    }
#else
    static void Equal(const ValueType* data, const ValueType* value, bool* mask, size_t count) {
        for (size_t i = 0; i < count; ++i) mask[i] = (data[i] == *value);
    }
    static void LessThan(const ValueType* data, const ValueType* value, bool* mask, size_t count) {
        for (size_t i = 0; i < count; ++i) mask[i] = (data[i] < *value);
    }
    static void LessEqual(const ValueType* data, const ValueType* value, bool* mask, size_t count) {
        for (size_t i = 0; i < count; ++i) mask[i] = (data[i] <= *value);
    }
#endif
};

class ColumnScanner {
    std::shared_ptr<Segment> segment_;
    size_t column_index_;
    zone_map::ZoneMapEntry zone_map_entry_;
    bloom_filter::BloomFilter bloom_filter_;
    zone_map::ZoneMap zone_map_;
    const void* filter_value_ = nullptr;
    PredicateType filter_pred_ = PredicateType::Equal;
    int64_t filter_int64_ = 0;
    
public:
    ColumnScanner(std::shared_ptr<Segment> seg, size_t col_idx) 
        : segment_(seg), column_index_(col_idx) {}
    
    void Initialize() {
        zone_map_ = segment_->GetZoneMap(column_index_);
        bloom_filter_ = segment_->GetBloomFilter(column_index_);
    }
    
    bool CanPrune(PredicateType pred, const void* value) const {
        return !zone_map_.ShouldScanBlock(0, segment_->ColumnType(column_index_), pred, value);
    }
    
    bool BloomFilterCheck(const void* value) const {
        if (!value) return true;
        DataType type = segment_->ColumnType(column_index_);
        switch (type) {
            case DataType::Int32: return bloom_filter_.MightContain(*static_cast<const int32_t*>(value));
            case DataType::Int64: return bloom_filter_.MightContain(*static_cast<const int64_t*>(value));
            case DataType::Float: return bloom_filter_.MightContain(*static_cast<const float*>(value));
            case DataType::Double: return bloom_filter_.MightContain(*static_cast<const double*>(value));
            default: return true;
        }
    }
    
    bool BloomFilterCheckInt64(int64_t value) const {
        return bloom_filter_.MightContain(value);
    }
    
    void ScanBlock(size_t /*block_idx*/, size_t offset, size_t count,
                   const void* filter_value, PredicateType pred,
                   BlockFilter& mask, void* /*output*/) const {
        if (count == 0) return;
        
        DataType type = segment_->ColumnType(column_index_);
        
        if (pred == PredicateType::IsNull) {
            mask.SetAll(false);
            return;
        }
        if (pred == PredicateType::IsNotNull) {
            mask.SetAll(true);
            return;
        }
        
        std::vector<uint8_t> block_data(count * TypeSize(type));
        segment_->GetColumnData(column_index_, offset, count, block_data.data());
        
        switch (type) {
            case DataType::Int32: {
                const int32_t* data = reinterpret_cast<const int32_t*>(block_data.data());
                int32_t value = *static_cast<const int32_t*>(filter_value);
                switch (pred) {
                    case PredicateType::Equal: VectorFilter<DataType::Int32>::Equal(data, &value, mask.Data(), count); break;
                    case PredicateType::NotEqual: {
                        VectorFilter<DataType::Int32>::Equal(data, &value, mask.Data(), count);
                        for (size_t i = 0; i < count; ++i) mask.Data()[i] = !mask.Data()[i];
                        break;
                    }
                    case PredicateType::LessThan: VectorFilter<DataType::Int32>::LessThan(data, &value, mask.Data(), count); break;
                    case PredicateType::LessEqual: VectorFilter<DataType::Int32>::LessEqual(data, &value, mask.Data(), count); break;
                    case PredicateType::GreaterThan: {
                        VectorFilter<DataType::Int32>::LessEqual(data, &value, mask.Data(), count);
                        for (size_t i = 0; i < count; ++i) mask.Data()[i] = !mask.Data()[i];
                        break;
                    }
                    case PredicateType::GreaterEqual: {
                        VectorFilter<DataType::Int32>::LessThan(data, &value, mask.Data(), count);
                        for (size_t i = 0; i < count; ++i) mask.Data()[i] = !mask.Data()[i];
                        break;
                    }
                    default: mask.SetAll(true); break;
                }
                break;
            }
            case DataType::Int64: {
                const int64_t* data = reinterpret_cast<const int64_t*>(block_data.data());
                int64_t value = *static_cast<const int64_t*>(filter_value);
                switch (pred) {
                    case PredicateType::Equal: VectorFilter<DataType::Int64>::Equal(data, &value, mask.Data(), count); break;
                    case PredicateType::NotEqual: {
                        VectorFilter<DataType::Int64>::Equal(data, &value, mask.Data(), count);
                        for (size_t i = 0; i < count; ++i) mask.Data()[i] = !mask.Data()[i];
                        break;
                    }
                    case PredicateType::LessThan: VectorFilter<DataType::Int64>::LessThan(data, &value, mask.Data(), count); break;
                    case PredicateType::LessEqual: VectorFilter<DataType::Int64>::LessEqual(data, &value, mask.Data(), count); break;
                    case PredicateType::GreaterThan: {
                        VectorFilter<DataType::Int64>::LessEqual(data, &value, mask.Data(), count);
                        for (size_t i = 0; i < count; ++i) mask.Data()[i] = !mask.Data()[i];
                        break;
                    }
                    case PredicateType::GreaterEqual: {
                        VectorFilter<DataType::Int64>::LessThan(data, &value, mask.Data(), count);
                        for (size_t i = 0; i < count; ++i) mask.Data()[i] = !mask.Data()[i];
                        break;
                    }
                    default: mask.SetAll(true); break;
                }
                break;
            }
            default: mask.SetAll(true); break;
        }
    }
    
    void Materialize(const bool* mask, size_t count) {
        if (!mask) return;
        size_t matched = 0;
        for (size_t i = 0; i < count; ++i) {
            if (mask[i]) ++matched;
        }
    }
    
    size_t Materialize(const bool* mask, size_t count, void* output) const {
        if (!mask || !output) return 0;
        DataType type = segment_->ColumnType(column_index_);
        size_t elem_size = TypeSize(type);
        size_t matched = 0;
        std::vector<uint8_t> block_data(count * elem_size);
        segment_->GetColumnData(column_index_, 0, count, block_data.data());
        for (size_t i = 0; i < count; ++i) {
            if (mask[i]) {
                std::memcpy(static_cast<uint8_t*>(output) + matched * elem_size,
                           block_data.data() + i * elem_size, elem_size);
                ++matched;
            }
        }
        return matched;
    }
    
    zone_map::ZoneMap& MutableZoneMap() { return zone_map_; }
    const bloom_filter::BloomFilter& GetBloomFilterRef() const { return bloom_filter_; }
    size_t ColumnIndex() const { return column_index_; }
};

class SegmentScanner {
    std::shared_ptr<Segment> segment_;
    std::vector<std::unique_ptr<ColumnScanner>> column_scanners_;
    ScanContext context_;
    ScanStats stats_;
    
public:
    SegmentScanner(std::shared_ptr<Segment> seg, const ScanContext& ctx)
        : segment_(seg), context_(ctx) {
        column_scanners_.resize(seg->NumColumns());
        for (size_t i = 0; i < seg->NumColumns(); ++i) {
            column_scanners_[i] = std::make_unique<ColumnScanner>(seg, i);
            column_scanners_[i]->Initialize();
        }
    }
    
    void Scan(const predicate::Predicate& predicate, 
              const std::function<void(const void**, size_t)>& output) {
        auto start = std::chrono::high_resolution_clock::now();
        
        size_t num_rows = segment_->NumRows();
        size_t num_blocks = (num_rows + context_.block_size - 1) / context_.block_size;
        BlockFilter block_mask;
        
        // Get filter info from the first column predicate for zone map pruning
        const auto& preds = predicate.ColumnPredicates();
        const auto& constants = predicate.Program().Constants();
        const auto& col_indices = predicate.Program().ColumnIndices();
        const size_t num_columns = segment_->NumColumns();
        predicate::BytecodeInterpreter interp(predicate.Program());
        
        for (size_t block = 0; block < num_blocks; ++block) {
            block_mask.Reset();
            stats_.blocks_scanned++;
            size_t block_offset = block * context_.block_size;
            size_t block_count = std::min(context_.block_size, num_rows - block_offset);
            stats_.rows_scanned += block_count;
            
            // Zone map pruning
            if (context_.enable_zone_map_pruning && !preds.empty() && !constants.empty()) {
                DataType type = segment_->ColumnType(col_indices[0]);
                bool should_scan = true;
                
                switch (type) {
                    case DataType::Int32: {
                        int32_t val = static_cast<int32_t>(constants[0]);
                        should_scan = column_scanners_[col_indices[0]]->MutableZoneMap().ShouldScanBlock(
                            block, type, preds[0], &val);
                        break;
                    }
                    case DataType::Int64: {
                        int64_t val = constants[0];
                        should_scan = column_scanners_[col_indices[0]]->MutableZoneMap().ShouldScanBlock(
                            block, type, preds[0], &val);
                        break;
                    }
                    default: should_scan = true; break;
                }
                
                if (!should_scan) {
                    stats_.blocks_pruned_zone_map++;
                    continue;
                }
            }
            
            // Bloom filter check
            if (context_.enable_bloom_filter && !preds.empty() && !constants.empty()) {
                size_t col = col_indices[0];
                if (col < column_scanners_.size() && preds[0] == PredicateType::Equal) {
                    int64_t val = constants[0];
                    if (!column_scanners_[col]->BloomFilterCheckInt64(val)) {
                        stats_.blocks_pruned_bloom++;
                        continue;
                    }
                }
            }
            
            // Scan each projected column
            // Evaluate the compiled predicate on every candidate row and
            // deliver surviving rows (one pointer per projected column) to
            // the output callback.
            std::vector<int64_t> row_values(num_columns);
            std::vector<const void*> row_ptrs(num_columns);

            for (size_t i = 0; i < block_count; ++i) {
                if (!block_mask.Data()[i]) continue;

                for (size_t c = 0; c < num_columns; ++c) {
                    segment_->GetColumnData(c, block_offset + i, 1, &row_values[c]);
                }
                for (size_t c = 0; c < num_columns; ++c) {
                    row_ptrs[c] = &row_values[c];
                }

                // Each row_ptrs slot holds exactly one widened value, so the
                // interpreter reads index 0 of every column buffer.
                if (interp.EvaluateRow(row_ptrs, 0)) {
                    output(row_ptrs.data(), 1);
                }
            }

            size_t matched = block_mask.CountTrue();
            stats_.rows_filtered += block_count - matched;
            stats_.rows_returned += matched;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        stats_.scan_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    const ScanStats& Stats() const { return stats_; }
};

class ParallelScanner {
    std::vector<std::shared_ptr<Segment>> segments_;
    ScanContext context_;
    ScanStats total_stats_;
    std::atomic<bool> cancelled_{false};
    
public:
    ParallelScanner(std::vector<std::shared_ptr<Segment>> segs, const ScanContext& ctx)
        : segments_(std::move(segs)), context_(ctx) {}
    
    void Scan(const predicate::Predicate& predicate, 
              const std::function<void(const void**, size_t)>& output) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::thread> threads;
        std::vector<ScanStats> thread_stats(context_.num_threads);
        
        for (size_t t = 0; t < context_.num_threads; ++t) {
            threads.emplace_back([this, t, &predicate, &output, &thread_stats]() {
                ScanThread(t, predicate, output, thread_stats[t]);
            });
        }
        
        for (auto& th : threads) th.join();
        
        for (const auto& ts : thread_stats) {
            total_stats_.blocks_scanned += ts.blocks_scanned;
            total_stats_.blocks_pruned_zone_map += ts.blocks_pruned_zone_map;
            total_stats_.blocks_pruned_bloom += ts.blocks_pruned_bloom;
            total_stats_.rows_scanned += ts.rows_scanned;
            total_stats_.rows_filtered += ts.rows_filtered;
            total_stats_.rows_returned += ts.rows_returned;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        total_stats_.scan_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    void Cancel() { cancelled_.store(true); }
    const ScanStats& Stats() const { return total_stats_; }
    
private:
    void ScanThread(size_t thread_id, const predicate::Predicate& predicate,
                    const std::function<void(const void**, size_t)>& output,
                    ScanStats& stats) {
        for (size_t i = thread_id; i < segments_.size(); i += context_.num_threads) {
            if (cancelled_.load()) break;
            
            SegmentScanner scanner(segments_[i], context_);
            scanner.Scan(predicate, [&](const void** cols, size_t count) {
                if (!cancelled_.load()) output(cols, count);
            });
            
            const auto& s = scanner.Stats();
            stats.blocks_scanned += s.blocks_scanned;
            stats.blocks_pruned_zone_map += s.blocks_pruned_zone_map;
            stats.blocks_pruned_bloom += s.blocks_pruned_bloom;
            stats.rows_scanned += s.rows_scanned;
            stats.rows_filtered += s.rows_filtered;
            stats.rows_returned += s.rows_returned;
        }
    }
};

class AggregateScanner {
    SegmentScanner scanner_;
    std::vector<double> sums_;
    std::vector<int64_t> counts_;
    std::vector<int64_t> mins_;
    std::vector<int64_t> maxs_;
    
public:
    AggregateScanner(std::shared_ptr<Segment> seg, const std::vector<size_t>& agg_columns)
        : scanner_(seg, ScanContext{}), agg_columns_(agg_columns) {
        sums_.assign(agg_columns.size(), 0);
        counts_.assign(agg_columns.size(), 0);
        mins_.assign(agg_columns.size(), INT64_MAX);
        maxs_.assign(agg_columns.size(), INT64_MIN);
    }
    
    void Scan(const predicate::Predicate& predicate) {
        scanner_.Scan(predicate, [&](const void** cols, size_t count) {
            for (size_t i = 0; i < agg_columns_.size(); ++i) {
                size_t col_idx = agg_columns_[i];
                const int64_t* data = static_cast<const int64_t*>(cols[col_idx]);
                for (size_t j = 0; j < count; ++j) {
                    if (data[j] != 0 || true) {
                        sums_[i] += data[j];
                        counts_[i]++;
                        mins_[i] = std::min(mins_[i], data[j]);
                        maxs_[i] = std::max(maxs_[i], data[j]);
                    }
                }
            }
        });
    }
    
    struct Result {
        double sum;
        int64_t count;
        int64_t min;
        int64_t max;
    };
    
    std::vector<Result> Results() const {
        std::vector<Result> results;
        for (size_t i = 0; i < agg_columns_.size(); ++i) {
            results.push_back({sums_[i], counts_[i], mins_[i], maxs_[i]});
        }
        return results;
    }
    
private:
    std::vector<size_t> agg_columns_;
};

} // namespace scanner
} // namespace columnar
