#pragma once

#include <columnar/types.h>
#include <columnar/encoding/encoding.h>
#include <columnar/column/column.h>
#include <columnar/zone_map.h>
#include <columnar/bloom_filter.h>
#include <vector>
#include <memory>
#include <cstdint>
#include <string>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <thread>

namespace columnar {

using column::ColumnBase;
using column::TypedColumn;
using column::ColumnFactory;

class Segment;

struct SegmentHeader {
    uint32_t magic = 0x434F4C53;
    uint32_t version = 1;
    uint32_t num_columns = 0;
    uint64_t num_rows = 0;
    uint64_t created_ts = 0;
    uint64_t checksum = 0;
    
    struct ColumnMeta {
        uint32_t offset;
        uint32_t size;
        DataType type;
        EncodingType encoding;
        ColumnStats stats;
        uint32_t zone_map_offset;
        uint32_t zone_map_size;
        uint32_t bloom_offset;
        uint32_t bloom_size;
    };
    
    std::vector<ColumnMeta> columns;
    uint32_t footer_offset = 0;
    uint32_t footer_size = 0;
};

class SegmentReader {
    std::string path_;
    std::ifstream file_;
    SegmentHeader header_;
    std::vector<uint8_t> footer_data_;
    std::vector<std::shared_ptr<ColumnBase>> cached_columns_;
    std::mutex cache_mutex_;
    
public:
    explicit SegmentReader(const std::string& path) : path_(path) {
        file_.open(path, std::ios::binary);
        if (!file_) throw std::runtime_error("Failed to open segment: " + path);
        ReadHeader();
    }
    
    ~SegmentReader() { file_.close(); }
    
    const SegmentHeader& Header() const { return header_; }
    uint64_t NumRows() const { return header_.num_rows; }
    uint32_t NumColumns() const { return header_.num_columns; }
    DataType ColumnType(size_t idx) const { return header_.columns[idx].type; }
    EncodingType ColumnEncoding(size_t idx) const { return header_.columns[idx].encoding; }
    const ColumnStats& GetColumnStats(size_t idx) const { return header_.columns[idx].stats; }
    
    std::shared_ptr<ColumnBase> GetColumn(size_t idx) {
        std::lock_guard lock(cache_mutex_);
        
        if (idx < cached_columns_.size() && cached_columns_[idx]) {
            return cached_columns_[idx];
        }
        
        if (cached_columns_.size() <= idx) cached_columns_.resize(idx + 1);
        
        const auto& col_meta = header_.columns[idx];
        file_.seekg(col_meta.offset);
        
        std::vector<uint8_t> encoded_data(col_meta.size);
        file_.read(reinterpret_cast<char*>(encoded_data.data()), col_meta.size);
        
        auto column = ColumnFactory::Create(col_meta.type);
        auto decoder = encoding::CreateDecoder(col_meta.encoding, col_meta.type);
        column->Decode(decoder.get(), encoded_data.data(), encoded_data.size());
        
        cached_columns_[idx] = std::shared_ptr<ColumnBase>(std::move(column));
        return cached_columns_[idx];
    }
    
    zone_map::ZoneMap GetZoneMap(size_t idx) {
        const auto& col_meta = header_.columns[idx];
        if (col_meta.zone_map_size == 0) return zone_map::ZoneMap();
        
        file_.seekg(col_meta.zone_map_offset);
        std::vector<uint8_t> data(col_meta.zone_map_size);
        file_.read(reinterpret_cast<char*>(data.data()), data.size());
        
        return zone_map::ZoneMap::Read(data.data(), data.size());
    }
    
    bloom_filter::BloomFilter GetBloomFilter(size_t idx) {
        const auto& col_meta = header_.columns[idx];
        if (col_meta.bloom_size == 0) return bloom_filter::BloomFilter();
        
        file_.seekg(col_meta.bloom_offset);
        std::vector<uint8_t> data(col_meta.bloom_size);
        file_.read(reinterpret_cast<char*>(data.data()), data.size());
        
        return bloom_filter::BloomFilter::Read(data.data(), data.size());
    }
    
private:
    void ReadHeader() {
        file_.seekg(0);
        file_.read(reinterpret_cast<char*>(&header_.magic), sizeof(header_.magic));
        file_.read(reinterpret_cast<char*>(&header_.version), sizeof(header_.version));
        file_.read(reinterpret_cast<char*>(&header_.num_columns), sizeof(header_.num_columns));
        file_.read(reinterpret_cast<char*>(&header_.num_rows), sizeof(header_.num_rows));
        file_.read(reinterpret_cast<char*>(&header_.created_ts), sizeof(header_.created_ts));
        file_.read(reinterpret_cast<char*>(&header_.checksum), sizeof(header_.checksum));
        
        header_.columns.resize(header_.num_columns);
        file_.read(reinterpret_cast<char*>(header_.columns.data()), 
                  header_.num_columns * sizeof(SegmentHeader::ColumnMeta));
    }
};

class SegmentWriter {
    std::string path_;
    std::ofstream file_;
    SegmentHeader header_;
    std::vector<encoding::Buffer> column_buffers_;
    std::vector<zone_map::ZoneMap> zone_maps_;
    std::vector<bloom_filter::BloomFilter> bloom_filters_;
    std::vector<ColumnStats> column_stats_;
    std::vector<EncodingType> column_encodings_;
    uint64_t row_count_ = 0;
    uint64_t timestamp_ = 0;
    
public:
    explicit SegmentWriter(const std::string& path);
    ~SegmentWriter();
    
    void SetTimestamp(uint64_t ts) { timestamp_ = ts; }
    
    template<DataType T>
    void AddColumn(const std::vector<typename TypeTraits<T>::Type>& values,
                   const std::vector<bool>& nulls,
                   EncodingType encoding = EncodingType::Plain) {
        using ValueType = typename TypeTraits<T>::Type;
        
        encoding::Buffer buffer;
        auto encoder = encoding::CreateEncoder(encoding, T);
        encoder->Encode(values.data(), values.size(), buffer);
        encoder->Finish(buffer);
        
        ColumnStats stats;
        stats.type = T;
        stats.total_count = values.size();
        stats.null_count = std::count(nulls.begin(), nulls.end(), true);
        stats.distinct_count = std::unordered_set<ValueType>(values.begin(), values.end()).size();
        
        if (!values.empty()) {
            stats.has_min_max = true;
            ValueType min_v = values[0], max_v = values[0];
            for (size_t i = 1; i < values.size(); ++i) {
                if (!nulls[i]) {
                    min_v = std::min(min_v, values[i]);
                    max_v = std::max(max_v, values[i]);
                }
            }
            stats.min_int = static_cast<int64_t>(min_v);
            stats.max_int = static_cast<int64_t>(max_v);
        }
        
        column_stats_.push_back(stats);
        column_encodings_.push_back(encoding);
        column_buffers_.push_back(std::move(buffer));
        header_.num_columns++;
        row_count_ = std::max(row_count_, values.size());
    }
    
    void AddZoneMap(size_t col_idx, const zone_map::ZoneMap& zm) {
        if (zone_maps_.size() <= col_idx) zone_maps_.resize(col_idx + 1);
        zone_maps_[col_idx] = zm;
    }
    
    void AddBloomFilter(size_t col_idx, const bloom_filter::BloomFilter& bf) {
        if (bloom_filters_.size() <= col_idx) bloom_filters_.resize(col_idx + 1);
        bloom_filters_[col_idx] = bf;
    }
    
    std::shared_ptr<Segment> Finish();
};

class Segment : public std::enable_shared_from_this<Segment> {
    std::string path_;
    SegmentHeader header_;
    std::unique_ptr<SegmentReader> reader_;
    mutable std::vector<std::shared_ptr<ColumnBase>> column_cache_;
    mutable std::mutex cache_mutex_;
    
public:
    Segment(const std::string& path, const SegmentHeader& header) 
        : path_(path), header_(header), reader_(std::make_unique<SegmentReader>(path)) {
        column_cache_.resize(header_.num_columns);
    }
    
    const std::string& Path() const { return path_; }
    const SegmentHeader& Header() const { return header_; }
    uint64_t NumRows() const { return header_.num_rows; }
    uint32_t NumColumns() const { return header_.num_columns; }
    DataType ColumnType(size_t idx) const { return header_.columns[idx].type; }
    EncodingType ColumnEncoding(size_t idx) const { return header_.columns[idx].encoding; }
    const ColumnStats& GetColumnStats(size_t idx) const { return header_.columns[idx].stats; }
    
    std::shared_ptr<ColumnBase> GetColumn(size_t idx) const {
        std::lock_guard lock(cache_mutex_);
        if (idx < column_cache_.size() && column_cache_[idx]) {
            return column_cache_[idx];
        }
        return reader_->GetColumn(idx);
    }
    
    template<DataType T>
    std::shared_ptr<TypedColumn<T>> GetColumn() const {
        return std::static_pointer_cast<TypedColumn<T>>(GetColumn(0));
    }
    
    zone_map::ZoneMap GetZoneMap(size_t idx) const {
        return reader_->GetZoneMap(idx);
    }
    
    bloom_filter::BloomFilter GetBloomFilter(size_t idx) const {
        return reader_->GetBloomFilter(idx);
    }
    
    std::unique_ptr<encoding::Decoder> GetDecoder(size_t idx) const {
        return encoding::CreateDecoder(header_.columns[idx].encoding, header_.columns[idx].type);
    }
    
    void PrefetchColumns(const std::vector<size_t>& columns) const {
        for (size_t col : columns) {
            std::thread([this, col] { GetColumn(col); }).detach();
        }
    }
    
    void GetColumnData(size_t col, size_t offset, size_t count, void* out) const {
        DataType type = header_.columns[col].type;
        EncodingType encoding = header_.columns[col].encoding;
        
        auto column = GetColumn(col);
        
        switch (type) {
            case DataType::Int32: {
                auto* typed = static_cast<TypedColumn<DataType::Int32>*>(column.get());
                std::memcpy(out, typed->Data() + offset, count * sizeof(int32_t));
                break;
            }
            case DataType::Int64: {
                auto* typed = static_cast<TypedColumn<DataType::Int64>*>(column.get());
                std::memcpy(out, typed->Data() + offset, count * sizeof(int64_t));
                break;
            }
            case DataType::Float: {
                auto* typed = static_cast<TypedColumn<DataType::Float>*>(column.get());
                std::memcpy(out, typed->Data() + offset, count * sizeof(float));
                break;
            }
            case DataType::Double: {
                auto* typed = static_cast<TypedColumn<DataType::Double>*>(column.get());
                std::memcpy(out, typed->Data() + offset, count * sizeof(double));
                break;
            }
            default:
                std::memset(out, 0, count * TypeSize(type));
                break;
        }
    }
    
    static std::shared_ptr<Segment> Open(const std::string& path) {
        SegmentReader reader(path);
        return std::make_shared<Segment>(path, reader.Header());
    }
    
    static std::shared_ptr<Segment> Create(const std::string& path, 
                                           const std::vector<ColumnStats>& /*stats*/,
                                           uint64_t timestamp = 0) {
        SegmentWriter writer(path);
        writer.SetTimestamp(timestamp);
        return writer.Finish();
    }
};

inline SegmentWriter::SegmentWriter(const std::string& path) : path_(path) {
    file_.open(path, std::ios::binary);
    if (!file_) throw std::runtime_error("Failed to create segment: " + path);
    header_.created_ts = timestamp_;
}

inline SegmentWriter::~SegmentWriter() {
    if (file_.is_open()) Finish();
}

inline std::shared_ptr<Segment> SegmentWriter::Finish() {
    header_.num_rows = row_count_;
    
    // Calculate header size: fixed fields + ColumnMeta array
    constexpr size_t kFixedHeaderSize = sizeof(uint32_t) * 3 + sizeof(uint64_t) * 3;
    size_t header_size = kFixedHeaderSize + header_.num_columns * sizeof(SegmentHeader::ColumnMeta);
    
    // Write placeholder header (will be overwritten later)
    std::vector<uint8_t> header_placeholder(header_size, 0);
    file_.write(reinterpret_cast<const char*>(header_placeholder.data()), header_size);
    
    // Write column data, zone maps, bloom filters
    for (size_t i = 0; i < column_buffers_.size(); ++i) {
        SegmentHeader::ColumnMeta meta{};
        meta.type = column_stats_[i].type;
        meta.encoding = column_encodings_[i];
        meta.stats = column_stats_[i];
        meta.zone_map_offset = 0;
        meta.zone_map_size = 0;
        meta.bloom_offset = 0;
        meta.bloom_size = 0;
        
        meta.offset = static_cast<uint32_t>(file_.tellp());
        meta.size = static_cast<uint32_t>(column_buffers_[i].Size());
        file_.write(reinterpret_cast<const char*>(column_buffers_[i].Data()), meta.size);
        
        if (i < zone_maps_.size() && zone_maps_[i].NumBlocks() > 0) {
            encoding::Buffer zm_buf;
            zone_maps_[i].Write(zm_buf);
            meta.zone_map_offset = static_cast<uint32_t>(file_.tellp());
            meta.zone_map_size = static_cast<uint32_t>(zm_buf.Size());
            file_.write(reinterpret_cast<const char*>(zm_buf.Data()), zm_buf.Size());
        }
        
        if (i < bloom_filters_.size() && bloom_filters_[i].Size() > 0) {
            encoding::Buffer bf_buf;
            bloom_filters_[i].Write(bf_buf);
            meta.bloom_offset = static_cast<uint32_t>(file_.tellp());
            meta.bloom_size = static_cast<uint32_t>(bf_buf.Size());
            file_.write(reinterpret_cast<const char*>(bf_buf.Data()), bf_buf.Size());
        }
        
        header_.columns.push_back(meta);
    }
    
    // Seek back to beginning and write real header
    file_.seekp(0);
    file_.write(reinterpret_cast<const char*>(&header_.magic), sizeof(header_.magic));
    file_.write(reinterpret_cast<const char*>(&header_.version), sizeof(header_.version));
    file_.write(reinterpret_cast<const char*>(&header_.num_columns), sizeof(header_.num_columns));
    file_.write(reinterpret_cast<const char*>(&header_.num_rows), sizeof(header_.num_rows));
    file_.write(reinterpret_cast<const char*>(&header_.created_ts), sizeof(header_.created_ts));
    file_.write(reinterpret_cast<const char*>(&header_.checksum), sizeof(header_.checksum));
    for (const auto& col_meta : header_.columns) {
        file_.write(reinterpret_cast<const char*>(&col_meta), sizeof(SegmentHeader::ColumnMeta));
    }
    
    file_.close();
    
    return std::make_shared<Segment>(path_, header_);
}

} // namespace columnar
