#pragma once

#include <columnar/types.h>
#include <columnar/column/column.h>
#include <vector>
#include <cstdint>
#include <memory>
#include <mutex>
#include <atomic>
#include <map>
#include <shared_mutex>
#include <functional>

namespace columnar {
namespace delta {

struct DeltaEntry {
    uint64_t row_id;
    uint64_t timestamp;
    bool is_delete;
    std::vector<uint8_t> data;
};

class DeltaStore {
    struct ColumnDelta {
        std::map<uint64_t, std::pair<uint64_t, std::vector<uint8_t>>> inserts;
        std::map<uint64_t, uint64_t> deletes;
        std::shared_mutex mutex;
        
        size_t MemoryUsage() const {
            size_t usage = 0;
            for (const auto& insert_entry : inserts) usage += insert_entry.second.second.size() + 16;
            usage += deletes.size() * 16;
            return usage;
        }
    };
    
    std::vector<std::unique_ptr<ColumnDelta>> columns_;
    size_t num_columns_;
    size_t flush_threshold_;
    std::atomic<size_t> pending_rows_{0};
    std::mutex flush_mutex_;
    std::atomic<uint64_t> global_timestamp_{0};
    std::atomic<bool> flushing_{false};
    
public:
    explicit DeltaStore(size_t num_columns, size_t flush_threshold = 100000)
        : num_columns_(num_columns), flush_threshold_(flush_threshold) {
        columns_.resize(num_columns);
        for (size_t i = 0; i < num_columns; ++i) {
            columns_[i] = std::make_unique<ColumnDelta>();
        }
    }
    
    void Insert(uint64_t row_id, const std::vector<std::vector<uint8_t>>& column_data, uint64_t timestamp = 0) {
        if (timestamp == 0) timestamp = ++global_timestamp_;
        
        for (size_t col = 0; col < num_columns_ && col < column_data.size(); ++col) {
            std::unique_lock lock(columns_[col]->mutex);
            columns_[col]->inserts[row_id] = {timestamp, column_data[col]};
        }
        pending_rows_.fetch_add(1, std::memory_order_relaxed);
    }
    
    void Delete(uint64_t row_id, uint64_t timestamp = 0) {
        if (timestamp == 0) timestamp = ++global_timestamp_;
        
        for (size_t col = 0; col < num_columns_; ++col) {
            std::unique_lock lock(columns_[col]->mutex);
            columns_[col]->deletes[row_id] = timestamp;
        }
    }
    
    bool Get(size_t col, uint64_t row_id, std::vector<uint8_t>& out_data, uint64_t& out_timestamp) const {
        if (col >= num_columns_) return false;
        
        std::shared_lock lock(columns_[col]->mutex);
        
        auto del_it = columns_[col]->deletes.find(row_id);
        if (del_it != columns_[col]->deletes.end()) {
            return false;
        }
        
        auto ins_it = columns_[col]->inserts.find(row_id);
        if (ins_it != columns_[col]->inserts.end()) {
            out_data = ins_it->second.second;
            out_timestamp = ins_it->second.first;
            return true;
        }
        
        return false;
    }
    
    void GetRange(size_t col, uint64_t start_row, uint64_t end_row, 
                  std::vector<std::pair<uint64_t, std::vector<uint8_t>>>& out) const {
        if (col >= num_columns_) return;
        
        std::shared_lock lock(columns_[col]->mutex);
        
        for (auto it = columns_[col]->inserts.lower_bound(start_row); 
             it != columns_[col]->inserts.end() && it->first < end_row; ++it) {
            auto del_it = columns_[col]->deletes.find(it->first);
            if (del_it == columns_[col]->deletes.end() || del_it->second < it->second.first) {
                out.emplace_back(it->first, it->second.second);
            }
        }
    }
    
    bool ShouldFlush() const {
        return pending_rows_.load(std::memory_order_relaxed) >= flush_threshold_;
    }
    
    size_t PendingRows() const {
        return pending_rows_.load(std::memory_order_relaxed);
    }
    
    size_t MemoryUsage() const {
        size_t total = 0;
        for (const auto& col : columns_) {
            std::shared_lock lock(col->mutex);
            total += col->MemoryUsage();
        }
        return total;
    }
    
    void BeginFlush() {
        flushing_.store(true, std::memory_order_release);
    }
    
    void EndFlush(size_t flushed_rows) {
        pending_rows_.fetch_sub(flushed_rows, std::memory_order_relaxed);
        flushing_.store(false, std::memory_order_release);
    }
    
    bool IsFlushing() const {
        return flushing_.load(std::memory_order_acquire);
    }
    
    uint64_t GlobalTimestamp() const {
        return global_timestamp_.load(std::memory_order_relaxed);
    }
    
    void Clear() {
        for (auto& col : columns_) {
            std::unique_lock lock(col->mutex);
            col->inserts.clear();
            col->deletes.clear();
        }
        pending_rows_.store(0, std::memory_order_relaxed);
    }
    
    template<typename Visitor>
    void VisitInserts(size_t col, Visitor&& visitor) const {
        if (col >= num_columns_) return;
        std::shared_lock lock(columns_[col]->mutex);
        for (const auto& [row_id, data] : columns_[col]->inserts) {
            visitor(row_id, data.first, data.second);
        }
    }
    
    template<typename Visitor>
    void VisitDeletes(size_t col, Visitor&& visitor) const {
        if (col >= num_columns_) return;
        std::shared_lock lock(columns_[col]->mutex);
        for (const auto& [row_id, ts] : columns_[col]->deletes) {
            visitor(row_id, ts);
        }
    }
};

class DeltaStoreManager {
    std::vector<std::unique_ptr<DeltaStore>> stores_;
    size_t num_columns_;
    size_t flush_threshold_;
    std::mutex manager_mutex_;
    std::atomic<size_t> active_store_{0};
    
public:
    DeltaStoreManager(size_t num_columns, size_t flush_threshold = 100000)
        : num_columns_(num_columns), flush_threshold_(flush_threshold) {
        stores_.push_back(std::make_unique<DeltaStore>(num_columns, flush_threshold));
    }
    
    DeltaStore* CurrentStore() {
        return stores_[active_store_.load(std::memory_order_relaxed)].get();
    }
    
    DeltaStore* GetStoreForFlush() {
        std::lock_guard lock(manager_mutex_);
        size_t current = active_store_.load();
        if (stores_[current]->ShouldFlush()) {
            if (stores_.size() == 1) {
                stores_.push_back(std::make_unique<DeltaStore>(num_columns_, flush_threshold_));
            }
            active_store_.store((current + 1) % stores_.size());
            stores_[current]->BeginFlush();
            return stores_[current].get();
        }
        return nullptr;
    }
    
    void FinishFlush(size_t store_idx, size_t flushed_rows) {
        if (store_idx < stores_.size()) {
            stores_[store_idx]->EndFlush(flushed_rows);
        }
    }
    
    void CompactStores() {
        std::lock_guard lock(manager_mutex_);
        for (size_t i = 0; i < stores_.size(); ++i) {
            if (!stores_[i]->IsFlushing() && stores_[i]->PendingRows() == 0) {
                stores_[i]->Clear();
            }
        }
    }
};

} // namespace delta
} // namespace columnar