#pragma once

#include <columnar/types.h>
#include <columnar/segment.h>
#include <columnar/delta_store.h>
#include <columnar/encoding/encoding.h>
#include <vector>
#include <cstdint>
#include <memory>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <numeric>

namespace columnar {
namespace merger {

struct MergeTask {
    std::vector<std::shared_ptr<Segment>> input_segments;
    std::shared_ptr<delta::DeltaStore> delta_store;
    std::string output_path;
    DataType column_type = DataType::Int32;
    EncodingType encoding_type = EncodingType::Plain;
    uint64_t min_timestamp = 0;
    uint64_t max_timestamp = 0;
    std::promise<std::shared_ptr<Segment>> promise;
};

struct MergeStats {
    size_t input_rows = 0;
    size_t output_rows = 0;
    size_t input_bytes = 0;
    size_t output_bytes = 0;
    double compression_ratio = 0.0;
    uint64_t merge_time_us = 0;
};

class ColumnMerger {
    DataType column_type_;
    EncodingType encoding_type_;
    size_t column_index_ = 0;
    
public:
    ColumnMerger(DataType type, EncodingType encoding) 
        : column_type_(type), encoding_type_(encoding) {}
    
    template<DataType T>
    void MergeColumn(const std::vector<std::shared_ptr<Segment>>& segments,
                     const std::shared_ptr<delta::DeltaStore>& delta,
                     std::vector<uint8_t>& output_data,
                     ColumnStats& out_stats) {
        using ValueType = typename TypeTraits<T>::Type;
        
        std::vector<ValueType> all_values;
        std::vector<uint64_t> all_timestamps;
        
        for (const auto& seg : segments) {
            size_t n = seg->NumRows();
            std::vector<ValueType> col_data(n);
            seg->GetColumnData(column_index_, 0, n, col_data.data());
            for (size_t i = 0; i < n; ++i) {
                all_values.push_back(col_data[i]);
                all_timestamps.push_back(seg->Header().created_ts);
            }
        }
        
        if (delta) {
            delta->VisitInserts(column_index_, [&](uint64_t /*row_id*/, uint64_t ts, const std::vector<uint8_t>& data) {
                ValueType value{};
                std::memcpy(&value, data.data(), std::min(data.size(), sizeof(ValueType)));
                all_values.push_back(value);
                all_timestamps.push_back(ts);
            });
        }
        
        std::vector<size_t> indices(all_values.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::stable_sort(indices.begin(), indices.end(), 
            [&](size_t a, size_t b) { return all_timestamps[a] < all_timestamps[b]; });
        
        std::vector<ValueType> sorted_values(all_values.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            sorted_values[i] = all_values[indices[i]];
        }
        
        encoding::Buffer buffer;
        auto encoder = encoding::CreateEncoder<EncodingType::Plain, T>();
        encoder->Encode(sorted_values.data(), sorted_values.size(), buffer);
        encoder->Finish(buffer);
        
        output_data.resize(buffer.Size());
        std::memcpy(output_data.data(), buffer.Data(), buffer.Size());
        
        out_stats.type = column_type_;
        out_stats.total_count = sorted_values.size();
        out_stats.null_count = 0;
        
        if (!sorted_values.empty()) {
            out_stats.has_min_max = true;
            out_stats.min_int = static_cast<int64_t>(sorted_values[0]);
            out_stats.max_int = static_cast<int64_t>(sorted_values[0]);
            for (size_t i = 1; i < sorted_values.size(); ++i) {
                int64_t v = static_cast<int64_t>(sorted_values[i]);
                if (v < out_stats.min_int) out_stats.min_int = v;
                if (v > out_stats.max_int) out_stats.max_int = v;
            }
        }
    }
    
    void SetColumnIndex(size_t idx) { column_index_ = idx; }
};

class SegmentMerger {
    std::vector<std::unique_ptr<ColumnMerger>> column_mergers_;
    size_t num_threads_;
    std::vector<std::thread> workers_;
    std::queue<MergeTask> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_{false};
    MergeStats last_stats_;
    
public:
    SegmentMerger(size_t num_columns, const std::vector<DataType>& types, 
                  const std::vector<EncodingType>& encodings, size_t num_threads = 4)
        : num_threads_(num_threads) {
        column_mergers_.reserve(num_columns);
        for (size_t i = 0; i < num_columns; ++i) {
            column_mergers_.push_back(std::make_unique<ColumnMerger>(types[i], encodings[i]));
            column_mergers_.back()->SetColumnIndex(i);
        }
        
        for (size_t i = 0; i < num_threads_; ++i) {
            workers_.emplace_back([this] { WorkerLoop(); });
        }
    }
    
    ~SegmentMerger() {
        stop_.store(true);
        queue_cv_.notify_all();
        for (auto& w : workers_) w.join();
    }
    
    std::future<std::shared_ptr<Segment>> SubmitMerge(MergeTask&& task) {
        auto future = task.promise.get_future();
        {
            std::lock_guard lock(queue_mutex_);
            task_queue_.push(std::move(task));
        }
        queue_cv_.notify_one();
        return future;
    }
    
    MergeStats LastStats() const { return last_stats_; }
    
private:
    void WorkerLoop() {
        while (!stop_.load()) {
            MergeTask task;
            {
                std::unique_lock lock(queue_mutex_);
                queue_cv_.wait(lock, [this] { return stop_.load() || !task_queue_.empty(); });
                if (stop_.load() && task_queue_.empty()) return;
                task = std::move(task_queue_.front());
                task_queue_.pop();
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            auto result = ExecuteMerge(std::move(task));
            auto end = std::chrono::high_resolution_clock::now();
            
            last_stats_.merge_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            task.promise.set_value(result);
        }
    }
    
    std::shared_ptr<Segment> ExecuteMerge(MergeTask&& task) {
        if (task.input_segments.empty()) {
            SegmentWriter writer(task.output_path);
            writer.SetTimestamp(task.max_timestamp);
            return writer.Finish();
        }
        if (task.input_segments.size() == 1) return task.input_segments[0];
        
        size_t num_columns = task.input_segments[0]->NumColumns();
        
        SegmentWriter writer(task.output_path);
        writer.SetTimestamp(task.max_timestamp);
        
        for (size_t col = 0; col < num_columns; ++col) {
            DataType type = task.input_segments[0]->ColumnType(col);
            EncodingType encoding = task.input_segments[0]->ColumnEncoding(col);
            
            std::vector<uint8_t> all_data;
            
            for (const auto& seg : task.input_segments) {
                size_t n = seg->NumRows();
                size_t elem_size = TypeSize(type);
                size_t offset = all_data.size();
                all_data.resize(offset + n * elem_size);
                seg->GetColumnData(col, 0, n, all_data.data() + offset);
            }
            
            ColumnStats stats;
            stats.type = type;
            stats.total_count = all_data.size() / TypeSize(type);
            stats.has_min_max = true;
            stats.min_int = INT64_MAX;
            stats.max_int = INT64_MIN;
            
            if (type == DataType::Int32) {
                const int32_t* data = reinterpret_cast<const int32_t*>(all_data.data());
                size_t count = all_data.size() / sizeof(int32_t);
                if (count > 0) {
                    stats.min_int = data[0];
                    stats.max_int = data[0];
                    for (size_t i = 1; i < count; ++i) {
                        stats.min_int = std::min(stats.min_int, static_cast<int64_t>(data[i]));
                        stats.max_int = std::max(stats.max_int, static_cast<int64_t>(data[i]));
                    }
                }
            } else if (type == DataType::Int64) {
                const int64_t* data = reinterpret_cast<const int64_t*>(all_data.data());
                size_t count = all_data.size() / sizeof(int64_t);
                if (count > 0) {
                    stats.min_int = data[0];
                    stats.max_int = data[0];
                    for (size_t i = 1; i < count; ++i) {
                        stats.min_int = std::min(stats.min_int, data[i]);
                        stats.max_int = std::max(stats.max_int, data[i]);
                    }
                }
            }
            
            switch (type) {
                case DataType::Int32: {
                    const int32_t* data = reinterpret_cast<const int32_t*>(all_data.data());
                    size_t count = all_data.size() / sizeof(int32_t);
                    std::vector<int32_t> values(data, data + count);
                    std::vector<bool> nulls(count, false);
                    writer.AddColumn<DataType::Int32>(values, nulls, encoding);
                    break;
                }
                case DataType::Int64: {
                    const int64_t* data = reinterpret_cast<const int64_t*>(all_data.data());
                    size_t count = all_data.size() / sizeof(int64_t);
                    std::vector<int64_t> values(data, data + count);
                    std::vector<bool> nulls(count, false);
                    writer.AddColumn<DataType::Int64>(values, nulls, encoding);
                    break;
                }
                case DataType::Float: {
                    const float* data = reinterpret_cast<const float*>(all_data.data());
                    size_t count = all_data.size() / sizeof(float);
                    std::vector<float> values(data, data + count);
                    std::vector<bool> nulls(count, false);
                    writer.AddColumn<DataType::Float>(values, nulls, encoding);
                    break;
                }
                case DataType::Double: {
                    const double* data = reinterpret_cast<const double*>(all_data.data());
                    size_t count = all_data.size() / sizeof(double);
                    std::vector<double> values(data, data + count);
                    std::vector<bool> nulls(count, false);
                    writer.AddColumn<DataType::Double>(values, nulls, encoding);
                    break;
                }
                default: break;
            }
        }
        
        return writer.Finish();
    }
};

class BackgroundMerger {
    std::unique_ptr<SegmentMerger> merger_;
    std::atomic<bool> running_{false};
    std::thread merger_thread_;
    std::function<void(const MergeStats&)> stats_callback_;
    
public:
    BackgroundMerger(size_t num_columns, const std::vector<DataType>& types,
                     const std::vector<EncodingType>& encodings, size_t threads = 2)
        : merger_(std::make_unique<SegmentMerger>(num_columns, types, encodings, threads)) {}
    
    void Start() {
        running_.store(true);
        merger_thread_ = std::thread([this] { RunLoop(); });
    }
    
    void Stop() {
        running_.store(false);
        if (merger_thread_.joinable()) merger_thread_.join();
    }
    
    void SetStatsCallback(std::function<void(const MergeStats&)> cb) {
        stats_callback_ = std::move(cb);
    }
    
    std::future<std::shared_ptr<Segment>> ScheduleMerge(MergeTask&& task) {
        return merger_->SubmitMerge(std::move(task));
    }
    
private:
    void RunLoop() {
        while (running_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
};

} // namespace merger
} // namespace columnar
