#pragma once

#include <columnar/types.h>
#include <columnar/encoding/encoding.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <immintrin.h>

namespace columnar {
namespace bloom_filter {

constexpr size_t kDefaultBits = 1 << 20;
constexpr size_t kDefaultHashes = 7;

class BloomFilter {
    std::vector<uint64_t> bits_;
    size_t num_hashes_;
    size_t num_bits_;
    
    static inline uint64_t MurmurHash64(uint64_t h, uint64_t seed) {
        h ^= seed;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return h;
    }
    
    static inline uint64_t HashCombine(uint64_t a, uint64_t b) {
        return a * 31 + b;
    }
    
public:
    BloomFilter() : num_hashes_(kDefaultHashes), num_bits_(kDefaultBits) {
        bits_.resize((num_bits_ + 63) / 64, 0);
    }
    
    explicit BloomFilter(size_t num_bits, size_t num_hashes = kDefaultHashes)
        : num_hashes_(num_hashes), num_bits_(num_bits) {
        bits_.resize((num_bits_ + 63) / 64, 0);
    }
    
    void Clear() {
        std::fill(bits_.begin(), bits_.end(), 0);
    }
    
    template<typename T>
    void Add(const T& value) {
        uint64_t val = 0;
        std::memcpy(&val, &value, sizeof(T));
        uint64_t h1 = MurmurHash64(val, 0x9e3779b97f4a7c15ULL);
        uint64_t h2 = MurmurHash64(val, 0xbf58476d1ce4e5b9ULL);
        
        for (size_t i = 0; i < num_hashes_; ++i) {
            uint64_t hash = h1 + i * h2;
            size_t bit = hash % num_bits_;
            bits_[bit / 64] |= (1ULL << (bit % 64));
        }
    }
    
    void AddString(const char* data, size_t len) {
        uint64_t h1 = MurmurHash64(reinterpret_cast<const uint64_t*>(data)[0], 0x9e3779b97f4a7c15ULL);
        uint64_t h2 = MurmurHash64(reinterpret_cast<const uint64_t*>(data)[0], 0xbf58476d1ce4e5b9ULL);
        
        for (size_t i = 0; i < num_hashes_; ++i) {
            uint64_t hash = h1 + i * h2;
            size_t bit = hash % num_bits_;
            bits_[bit / 64] |= (1ULL << (bit % 64));
        }
    }
    
    template<typename T>
    void AddBatch(const T* values, size_t count) {
        for (size_t i = 0; i < count; ++i) Add(values[i]);
    }
    
    template<typename T>
    bool MightContain(const T& value) const {
        uint64_t val = 0;
        std::memcpy(&val, &value, sizeof(T));
        uint64_t h1 = MurmurHash64(val, 0x9e3779b97f4a7c15ULL);
        uint64_t h2 = MurmurHash64(val, 0xbf58476d1ce4e5b9ULL);
        
        for (size_t i = 0; i < num_hashes_; ++i) {
            uint64_t hash = h1 + i * h2;
            size_t bit = hash % num_bits_;
            if ((bits_[bit / 64] & (1ULL << (bit % 64))) == 0) return false;
        }
        return true;
    }
    
    bool MightContainString(const char* data, size_t len) const {
        uint64_t h1 = MurmurHash64(reinterpret_cast<const uint64_t*>(data)[0], 0x9e3779b97f4a7c15ULL);
        uint64_t h2 = MurmurHash64(reinterpret_cast<const uint64_t*>(data)[0], 0xbf58476d1ce4e5b9ULL);
        
        for (size_t i = 0; i < num_hashes_; ++i) {
            uint64_t hash = h1 + i * h2;
            size_t bit = hash % num_bits_;
            if ((bits_[bit / 64] & (1ULL << (bit % 64))) == 0) return false;
        }
        return true;
    }
    
    template<typename T>
    void MightContainBatch(const T* values, size_t count, bool* results) const {
        for (size_t i = 0; i < count; ++i) {
            results[i] = MightContain(values[i]);
        }
    }
    
    void MightContainBatchAvx512(const int64_t* values, size_t count, bool* results) const {
#ifdef __AVX512F__
        size_t i = 0;
        for (; i + 7 < count; i += 8) {
            alignas(64) int64_t tmp_vals[8];
            _mm512_storeu_si512(tmp_vals, _mm512_loadu_si512(reinterpret_cast<const __m512i*>(values + i)));
            
            __mmask8 all_match = 0xFF;
            
            for (size_t h = 0; h < num_hashes_; ++h) {
                for (size_t j = 0; j < 8; ++j) {
                    uint64_t hash = MurmurHash64(tmp_vals[j], h * 0x9e3779b97f4a7c15ULL);
                    size_t bit = hash % num_bits_;
                    if ((bits_[bit / 64] & (1ULL << (bit % 64))) == 0) {
                        all_match &= ~(1ULL << j);
                    }
                }
                if (all_match == 0) break;
            }
            
            for (size_t j = 0; j < 8; ++j) {
                results[i + j] = (all_match >> j) & 1;
            }
        }
        
        for (; i < count; ++i) {
            results[i] = MightContain(values[i]);
        }
#else
        for (size_t i = 0; i < count; ++i) {
            results[i] = MightContain(values[i]);
        }
#endif
    }
    
    void Merge(const BloomFilter& other) {
        if (num_bits_ != other.num_bits_ || num_hashes_ != other.num_hashes_) return;
        for (size_t i = 0; i < bits_.size(); ++i) {
            bits_[i] |= other.bits_[i];
        }
    }
    
    double FalsePositiveRate() const {
        size_t bits_set = 0;
        for (uint64_t w : bits_) bits_set += __builtin_popcountll(w);
        double p = static_cast<double>(bits_set) / num_bits_;
        return std::pow(p, num_hashes_);
    }
    
    size_t BitsSet() const {
        size_t count = 0;
        for (uint64_t w : bits_) count += __builtin_popcountll(w);
        return count;
    }
    
    size_t NumBits() const { return num_bits_; }
    size_t NumHashes() const { return num_hashes_; }
    const uint64_t* Data() const { return bits_.data(); }
    uint64_t* Data() { return bits_.data(); }
    size_t Size() const { return bits_.size() * sizeof(uint64_t); }
    
    void Write(encoding::Buffer& out) const {
        out.Append(num_bits_);
        out.Append(num_hashes_);
        out.AppendArray(bits_.data(), bits_.size());
    }
    
    static BloomFilter Read(const uint8_t* data, size_t size) {
        BloomFilter bf;
        const uint8_t* ptr = data;
        
        std::memcpy(&bf.num_bits_, ptr, 8); ptr += 8;
        std::memcpy(&bf.num_hashes_, ptr, 8); ptr += 8;
        
        size_t num_words = (bf.num_bits_ + 63) / 64;
        bf.bits_.resize(num_words);
        std::memcpy(bf.bits_.data(), ptr, num_words * 8);
        
        return bf;
    }
};

class BlockedBloomFilter {
    std::vector<BloomFilter> blocks_;
    size_t block_size_;
    size_t num_hashes_;
    size_t bits_per_block_;
    
public:
    BlockedBloomFilter(size_t total_bits, size_t block_size, size_t num_hashes = kDefaultHashes)
        : block_size_(block_size), num_hashes_(num_hashes), bits_per_block_(total_bits) {
        size_t num_blocks = (total_bits + block_size - 1) / block_size;
        blocks_.reserve(num_blocks);
        for (size_t i = 0; i < num_blocks; ++i) {
            blocks_.emplace_back(bits_per_block_, num_hashes_);
        }
    }
    
    template<typename T>
    void Add(const T& value, size_t block_idx) {
        if (block_idx < blocks_.size()) blocks_[block_idx].Add(value);
    }
    
    template<typename T>
    bool MightContain(const T& value, size_t block_idx) const {
        if (block_idx >= blocks_.size()) return false;
        return blocks_[block_idx].MightContain(value);
    }
    
    template<typename T>
    void MightContainBatch(const T* values, size_t count, size_t block_idx, bool* results) const {
        if (block_idx >= blocks_.size()) {
            std::fill_n(results, count, false);
            return;
        }
        blocks_[block_idx].MightContainBatch(values, count, results);
    }
    
    const BloomFilter& Block(size_t idx) const { return blocks_[idx]; }
    size_t NumBlocks() const { return blocks_.size(); }
};

} // namespace bloom_filter
} // namespace columnar
