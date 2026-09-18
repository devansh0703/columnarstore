#pragma once

#include <columnar/types.h>
#include <columnar/encoding/encoding.h>
#include <vector>
#include <memory>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

namespace columnar {
namespace column {

constexpr size_t kVectorSize = 64;
constexpr size_t kAvx512Width = 16;
constexpr size_t kAvx2Width = 8;
constexpr size_t kSseWidth = 4;

template<DataType T>
struct VectorTraits;

template<>
struct VectorTraits<DataType::Int32> {
#ifdef __AVX512F__
    using Vec = __m512i;
    using Scalar = int32_t;
    static constexpr size_t Width = kAvx512Width;
    static Vec Load(const void* ptr) { return _mm512_loadu_si512(ptr); }
    static void Store(void* ptr, Vec v) { _mm512_storeu_si512(ptr, v); }
    static Vec Set1(int32_t v) { return _mm512_set1_epi32(v); }
    static __mmask16 CmpEq(Vec a, Vec b) { return _mm512_cmpeq_epi32_mask(a, b); }
    static __mmask16 CmpLt(Vec a, Vec b) { return _mm512_cmplt_epi32_mask(a, b); }
    static __mmask16 CmpLe(Vec a, Vec b) { return _mm512_cmple_epi32_mask(a, b); }
    static Vec Add(Vec a, Vec b) { return _mm512_add_epi32(a, b); }
    static Vec Sub(Vec a, Vec b) { return _mm512_sub_epi32(a, b); }
    static Vec Mul(Vec a, Vec b) { return _mm512_mullo_epi32(a, b); }
    static Vec And(Vec a, Vec b) { return _mm512_and_si512(a, b); }
    static Vec Or(Vec a, Vec b) { return _mm512_or_si512(a, b); }
    static Vec Xor(Vec a, Vec b) { return _mm512_xor_si512(a, b); }
    static Vec Not(Vec a) { return _mm512_xor_si512(a, _mm512_set1_epi32(-1)); }
#else
    using Vec = int32_t;
    using Scalar = int32_t;
    static constexpr size_t Width = 1;
    static Vec Load(const void* ptr) { return *static_cast<const int32_t*>(ptr); }
    static void Store(void* ptr, Vec v) { *static_cast<int32_t*>(ptr) = v; }
    static Vec Set1(int32_t v) { return v; }
    static int32_t CmpEq(Vec a, Vec b) { return (a == b) ? static_cast<int32_t>(-1) : 0; }
    static int32_t CmpLt(Vec a, Vec b) { return (a < b) ? static_cast<int32_t>(-1) : 0; }
    static int32_t CmpLe(Vec a, Vec b) { return (a <= b) ? static_cast<int32_t>(-1) : 0; }
    static Vec Add(Vec a, Vec b) { return a + b; }
    static Vec Sub(Vec a, Vec b) { return a - b; }
    static Vec Mul(Vec a, Vec b) { return a * b; }
    static Vec And(Vec a, Vec b) { return a & b; }
    static Vec Or(Vec a, Vec b) { return a | b; }
    static Vec Xor(Vec a, Vec b) { return a ^ b; }
    static Vec Not(Vec a) { return ~a; }
#endif
};

template<>
struct VectorTraits<DataType::Int64> {
#ifdef __AVX512F__
    using Vec = __m512i;
    using Scalar = int64_t;
    static constexpr size_t Width = kAvx512Width / 2;
    static Vec Load(const void* ptr) { return _mm512_loadu_si512(ptr); }
    static void Store(void* ptr, Vec v) { _mm512_storeu_si512(ptr, v); }
    static Vec Set1(int64_t v) { return _mm512_set1_epi64(v); }
    static __mmask8 CmpEq(Vec a, Vec b) { return _mm512_cmpeq_epi64_mask(a, b); }
    static __mmask8 CmpLt(Vec a, Vec b) { return _mm512_cmplt_epi64_mask(a, b); }
    static __mmask8 CmpLe(Vec a, Vec b) { return _mm512_cmple_epi64_mask(a, b); }
    static Vec Add(Vec a, Vec b) { return _mm512_add_epi64(a, b); }
    static Vec Sub(Vec a, Vec b) { return _mm512_sub_epi64(a, b); }
    static Vec Mul(Vec a, Vec b) { return _mm512_mullo_epi64(a, b); }
    static Vec And(Vec a, Vec b) { return _mm512_and_si512(a, b); }
    static Vec Or(Vec a, Vec b) { return _mm512_or_si512(a, b); }
    static Vec Xor(Vec a, Vec b) { return _mm512_xor_si512(a, b); }
    static Vec Not(Vec a) { return _mm512_xor_si512(a, _mm512_set1_epi64(-1)); }
#else
    using Vec = int64_t;
    using Scalar = int64_t;
    static constexpr size_t Width = 1;
    static Vec Load(const void* ptr) { return *static_cast<const int64_t*>(ptr); }
    static void Store(void* ptr, Vec v) { *static_cast<int64_t*>(ptr) = v; }
    static Vec Set1(int64_t v) { return v; }
    static int64_t CmpEq(Vec a, Vec b) { return (a == b) ? static_cast<int64_t>(-1) : 0; }
    static int64_t CmpLt(Vec a, Vec b) { return (a < b) ? static_cast<int64_t>(-1) : 0; }
    static int64_t CmpLe(Vec a, Vec b) { return (a <= b) ? static_cast<int64_t>(-1) : 0; }
    static Vec Add(Vec a, Vec b) { return a + b; }
    static Vec Sub(Vec a, Vec b) { return a - b; }
    static Vec Mul(Vec a, Vec b) { return a * b; }
    static Vec And(Vec a, Vec b) { return a & b; }
    static Vec Or(Vec a, Vec b) { return a | b; }
    static Vec Xor(Vec a, Vec b) { return a ^ b; }
    static Vec Not(Vec a) { return ~a; }
#endif
};

template<>
struct VectorTraits<DataType::Float> {
#ifdef __AVX512F__
    using Vec = __m512;
    using Scalar = float;
    static constexpr size_t Width = kAvx512Width;
    static Vec Load(const void* ptr) { return _mm512_loadu_ps(static_cast<const float*>(ptr)); }
    static void Store(void* ptr, Vec v) { _mm512_storeu_ps(static_cast<float*>(ptr), v); }
    static Vec Set1(float v) { return _mm512_set1_ps(v); }
    static __mmask16 CmpEq(Vec a, Vec b) { return _mm512_cmpeq_ps_mask(a, b); }
    static __mmask16 CmpLt(Vec a, Vec b) { return _mm512_cmplt_ps_mask(a, b); }
    static __mmask16 CmpLe(Vec a, Vec b) { return _mm512_cmple_ps_mask(a, b); }
    static Vec Add(Vec a, Vec b) { return _mm512_add_ps(a, b); }
    static Vec Sub(Vec a, Vec b) { return _mm512_sub_ps(a, b); }
    static Vec Mul(Vec a, Vec b) { return _mm512_mul_ps(a, b); }
    static Vec Div(Vec a, Vec b) { return _mm512_div_ps(a, b); }
#else
    using Vec = float;
    using Scalar = float;
    static constexpr size_t Width = 1;
    static Vec Load(const void* ptr) { return *static_cast<const float*>(ptr); }
    static void Store(void* ptr, Vec v) { *static_cast<float*>(ptr) = v; }
    static Vec Set1(float v) { return v; }
    static int32_t CmpEq(Vec a, Vec b) { return (a == b) ? static_cast<int32_t>(-1) : 0; }
    static int32_t CmpLt(Vec a, Vec b) { return (a < b) ? static_cast<int32_t>(-1) : 0; }
    static int32_t CmpLe(Vec a, Vec b) { return (a <= b) ? static_cast<int32_t>(-1) : 0; }
    static Vec Add(Vec a, Vec b) { return a + b; }
    static Vec Sub(Vec a, Vec b) { return a - b; }
    static Vec Mul(Vec a, Vec b) { return a * b; }
    static Vec Div(Vec a, Vec b) { return a / b; }
#endif
};

template<>
struct VectorTraits<DataType::Double> {
#ifdef __AVX512F__
    using Vec = __m512d;
    using Scalar = double;
    static constexpr size_t Width = kAvx512Width / 2;
    static Vec Load(const void* ptr) { return _mm512_loadu_pd(static_cast<const double*>(ptr)); }
    static void Store(void* ptr, Vec v) { _mm512_storeu_pd(static_cast<double*>(ptr), v); }
    static Vec Set1(double v) { return _mm512_set1_pd(v); }
    static __mmask8 CmpEq(Vec a, Vec b) { return _mm512_cmpeq_pd_mask(a, b); }
    static __mmask8 CmpLt(Vec a, Vec b) { return _mm512_cmplt_pd_mask(a, b); }
    static __mmask8 CmpLe(Vec a, Vec b) { return _mm512_cmple_pd_mask(a, b); }
    static Vec Add(Vec a, Vec b) { return _mm512_add_pd(a, b); }
    static Vec Sub(Vec a, Vec b) { return _mm512_sub_pd(a, b); }
    static Vec Mul(Vec a, Vec b) { return _mm512_mul_pd(a, b); }
    static Vec Div(Vec a, Vec b) { return _mm512_div_pd(a, b); }
#else
    using Vec = double;
    using Scalar = double;
    static constexpr size_t Width = 1;
    static Vec Load(const void* ptr) { return *static_cast<const double*>(ptr); }
    static void Store(void* ptr, Vec v) { *static_cast<double*>(ptr) = v; }
    static Vec Set1(double v) { return v; }
    static int32_t CmpEq(Vec a, Vec b) { return (a == b) ? static_cast<int32_t>(-1) : 0; }
    static int32_t CmpLt(Vec a, Vec b) { return (a < b) ? static_cast<int32_t>(-1) : 0; }
    static int32_t CmpLe(Vec a, Vec b) { return (a <= b) ? static_cast<int32_t>(-1) : 0; }
    static Vec Add(Vec a, Vec b) { return a + b; }
    static Vec Sub(Vec a, Vec b) { return a - b; }
    static Vec Mul(Vec a, Vec b) { return a * b; }
    static Vec Div(Vec a, Vec b) { return a / b; }
#endif
};

template<DataType T>
class VectorOps {
    using Traits = VectorTraits<T>;
    using Vec = typename Traits::Vec;
    using Scalar = typename Traits::Scalar;
    
public:
    static void Add(const Scalar* a, const Scalar* b, Scalar* out, size_t n) {
        for (size_t i = 0; i + Traits::Width <= n; i += Traits::Width) {
            Vec va = Traits::Load(a + i);
            Vec vb = Traits::Load(b + i);
            Traits::Store(out + i, Traits::Add(va, vb));
        }
        for (size_t i = n - n % Traits::Width; i < n; ++i) out[i] = a[i] + b[i];
    }
    
    static void Sub(const Scalar* a, const Scalar* b, Scalar* out, size_t n) {
        for (size_t i = 0; i + Traits::Width <= n; i += Traits::Width) {
            Vec va = Traits::Load(a + i);
            Vec vb = Traits::Load(b + i);
            Traits::Store(out + i, Traits::Sub(va, vb));
        }
        for (size_t i = n - n % Traits::Width; i < n; ++i) out[i] = a[i] - b[i];
    }
    
    static void Mul(const Scalar* a, const Scalar* b, Scalar* out, size_t n) {
        for (size_t i = 0; i + Traits::Width <= n; i += Traits::Width) {
            Vec va = Traits::Load(a + i);
            Vec vb = Traits::Load(b + i);
            Traits::Store(out + i, Traits::Mul(va, vb));
        }
        for (size_t i = n - n % Traits::Width; i < n; ++i) out[i] = a[i] * b[i];
    }
    
    template<PredicateType Pred>
    static void Filter(const Scalar* data, const Scalar* value, bool* mask, size_t n) {
        Vec vval = Traits::Set1(*value);
        for (size_t i = 0; i + Traits::Width <= n; i += Traits::Width) {
            Vec vdata = Traits::Load(data + i);
            auto m = Traits::CmpEq(vdata, vval);
            if constexpr (Pred == PredicateType::LessThan) m = Traits::CmpLt(vdata, vval);
            else if constexpr (Pred == PredicateType::LessEqual) m = Traits::CmpLe(vdata, vval);
            else if constexpr (Pred == PredicateType::GreaterThan) m = Traits::CmpLt(vval, vdata);
            else if constexpr (Pred == PredicateType::GreaterEqual) m = Traits::CmpLe(vval, vdata);
            else if constexpr (Pred == PredicateType::NotEqual) m = ~Traits::CmpEq(vdata, vval);
            
            for (size_t j = 0; j < Traits::Width; ++j) {
                mask[i + j] = (m >> j) & 1;
            }
        }
        for (size_t i = n - n % Traits::Width; i < n; ++i) {
            bool match = false;
            if constexpr (Pred == PredicateType::Equal) match = data[i] == *value;
            else if constexpr (Pred == PredicateType::LessThan) match = data[i] < *value;
            else if constexpr (Pred == PredicateType::LessEqual) match = data[i] <= *value;
            else if constexpr (Pred == PredicateType::GreaterThan) match = data[i] > *value;
            else if constexpr (Pred == PredicateType::GreaterEqual) match = data[i] >= *value;
            else if constexpr (Pred == PredicateType::NotEqual) match = data[i] != *value;
            mask[i] = match;
        }
    }
    
    static Scalar Sum(const Scalar* data, size_t n) {
#if defined(__AVX512F__) && defined(__GNUC__) && !defined(__clang__)
        // GCC warns about intrinsic vector types used as template arguments here
        // (__m512i etc. carry ABI attributes that are ignored in template deduction).
        // The is_same_v branches below dispatch correctly regardless.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
        Scalar sum = 0;
        for (size_t i = 0; i + Traits::Width <= n; i += Traits::Width) {
            Vec v = Traits::Load(data + i);
#ifdef __AVX512F__
            if constexpr (std::is_same_v<Vec, __m512>) {
                sum += _mm512_reduce_add_ps(v);
            } else if constexpr (std::is_same_v<Vec, __m512d>) {
                sum += _mm512_reduce_add_pd(v);
            } else {
#endif
                alignas(64) Scalar tmp[Traits::Width];
                Traits::Store(tmp, v);
                for (size_t j = 0; j < Traits::Width; ++j) sum += tmp[j];
#ifdef __AVX512F__
            }
#endif
        }
        for (size_t i = n - n % Traits::Width; i < n; ++i) sum += data[i];
        return sum;
#if defined(__AVX512F__) && defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
    }
    
    static Scalar Min(const Scalar* data, size_t n) {
#if defined(__AVX512F__) && defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
        Scalar min_val = data[0];
        for (size_t i = 0; i + Traits::Width <= n; i += Traits::Width) {
            Vec v = Traits::Load(data + i);
#ifdef __AVX512F__
            if constexpr (std::is_same_v<Vec, __m512>) {
                min_val = std::min(min_val, _mm512_reduce_min_ps(v));
            } else if constexpr (std::is_same_v<Vec, __m512d>) {
                min_val = std::min(min_val, _mm512_reduce_min_pd(v));
            } else {
#endif
                alignas(64) Scalar tmp[Traits::Width];
                Traits::Store(tmp, v);
                for (size_t j = 0; j < Traits::Width; ++j) min_val = std::min(min_val, tmp[j]);
#ifdef __AVX512F__
            }
#endif
        }
        for (size_t i = n - n % Traits::Width; i < n; ++i) min_val = std::min(min_val, data[i]);
        return min_val;
#if defined(__AVX512F__) && defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
    }
    
    static Scalar Max(const Scalar* data, size_t n) {
#if defined(__AVX512F__) && defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
        Scalar max_val = data[0];
        for (size_t i = 0; i + Traits::Width <= n; i += Traits::Width) {
            Vec v = Traits::Load(data + i);
#ifdef __AVX512F__
            if constexpr (std::is_same_v<Vec, __m512>) {
                max_val = std::max(max_val, _mm512_reduce_max_ps(v));
            } else if constexpr (std::is_same_v<Vec, __m512d>) {
                max_val = std::max(max_val, _mm512_reduce_max_pd(v));
            } else {
#endif
                alignas(64) Scalar tmp[Traits::Width];
                Traits::Store(tmp, v);
                for (size_t j = 0; j < Traits::Width; ++j) max_val = std::max(max_val, tmp[j]);
#ifdef __AVX512F__
            }
#endif
        }
        for (size_t i = n - n % Traits::Width; i < n; ++i) max_val = std::max(max_val, data[i]);
        return max_val;
#if defined(__AVX512F__) && defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
    }
};

template<DataType T>
class ColumnVector {
    using Native = NativeType<T>;
    
public:
    ColumnVector() = default;
    explicit ColumnVector(size_t capacity) : data_(capacity), nulls_(capacity, 0), size_(0) {}
    
    void Reserve(size_t capacity) {
        data_.reserve(capacity);
        nulls_.reserve(capacity);
    }
    
    void Resize(size_t size) {
        data_.resize(size);
        nulls_.resize(size);
        size_ = size;
    }
    
    void Append(const Native& value, bool is_null = false) {
        data_.push_back(value);
        nulls_.push_back(is_null ? 1 : 0);
        ++size_;
    }
    
    void Append(const Native* values, const uint8_t* nulls, size_t count) {
        data_.insert(data_.end(), values, values + count);
        if (nulls) {
            for (size_t i = 0; i < count; ++i) nulls_.push_back(nulls[i]);
        } else {
            nulls_.insert(nulls_.end(), count, 0);
        }
        size_ += count;
    }
    
    void AppendNulls(size_t count) {
        data_.insert(data_.end(), count, Native{});
        nulls_.insert(nulls_.end(), count, 1);
        size_ += count;
    }
    
    Native& operator[](size_t i) { return data_[i]; }
    const Native& operator[](size_t i) const { return data_[i]; }
    
    bool IsNull(size_t i) const { return nulls_[i]; }
    void SetNull(size_t i, bool null = true) { nulls_[i] = null ? 1 : 0; }
    
    Native* Data() { return data_.data(); }
    const Native* Data() const { return data_.data(); }
    uint8_t* Nulls() { return nulls_.data(); }
    const uint8_t* Nulls() const { return nulls_.data(); }
    
    size_t Size() const { return size_; }
    size_t Capacity() const { return data_.capacity(); }
    
    void Clear() {
        data_.clear();
        nulls_.clear();
        size_ = 0;
    }
    
    void Swap(ColumnVector& other) {
        data_.swap(other.data_);
        nulls_.swap(other.nulls_);
        std::swap(size_, other.size_);
    }
    
    template<PredicateType Pred>
    void Filter(const Native* value, bool* mask) const {
        VectorOps<T>::template Filter<Pred>(data_.data(), value, mask, size_);
    }
    
    void FilterNulls(bool* mask) const {
        for (size_t i = 0; i < size_; ++i) mask[i] = !nulls_[i];
    }
    
    void FilterNotNulls(bool* mask) const {
        for (size_t i = 0; i < size_; ++i) mask[i] = nulls_[i];
    }
    
private:
    std::vector<Native> data_;
    std::vector<uint8_t> nulls_;
    size_t size_ = 0;
};

template<>
class ColumnVector<DataType::Boolean> {
    using Native = bool;
    
public:
    ColumnVector() = default;
    explicit ColumnVector(size_t capacity) : data_(capacity), nulls_(capacity, 0), size_(0) {}
    
    void Reserve(size_t capacity) {
        data_.reserve(capacity);
        nulls_.reserve(capacity);
    }
    
    void Resize(size_t size) {
        data_.resize(size);
        nulls_.resize(size);
        size_ = size;
    }
    
    void Append(bool value, bool is_null = false) {
        data_.push_back(value);
        nulls_.push_back(is_null ? 1 : 0);
        ++size_;
    }
    
    void Append(const bool* values, const uint8_t* nulls, size_t count) {
        data_.insert(data_.end(), values, values + count);
        if (nulls) {
            for (size_t i = 0; i < count; ++i) nulls_.push_back(nulls[i]);
        } else {
            nulls_.insert(nulls_.end(), count, 0);
        }
        size_ += count;
    }
    
    void AppendNulls(size_t count) {
        data_.insert(data_.end(), count, false);
        nulls_.insert(nulls_.end(), count, 1);
        size_ += count;
    }
    
    bool& operator[](size_t i) { return *reinterpret_cast<bool*>(&data_[i]); }
    bool operator[](size_t i) const { return data_[i]; }
    
    bool IsNull(size_t i) const { return nulls_[i]; }
    void SetNull(size_t i, bool null = true) { nulls_[i] = null ? 1 : 0; }
    
    bool* Data() { return reinterpret_cast<bool*>(data_.data()); }
    const bool* Data() const { return reinterpret_cast<const bool*>(data_.data()); }
    uint8_t* Nulls() { return nulls_.data(); }
    const uint8_t* Nulls() const { return nulls_.data(); }
    
    size_t Size() const { return size_; }
    size_t Capacity() const { return data_.capacity(); }
    
    void Clear() {
        data_.clear();
        nulls_.clear();
        size_ = 0;
    }
    
    void Swap(ColumnVector& other) {
        data_.swap(other.data_);
        nulls_.swap(other.nulls_);
        std::swap(size_, other.size_);
    }
    
    template<PredicateType Pred>
    void Filter(const bool* value, bool* mask) const {
        for (size_t i = 0; i < size_; ++i) {
            bool match = false;
            if constexpr (Pred == PredicateType::Equal) match = data_[i] == *value;
            else if constexpr (Pred == PredicateType::NotEqual) match = data_[i] != *value;
            else if constexpr (Pred == PredicateType::IsNull) match = nulls_[i];
            else if constexpr (Pred == PredicateType::IsNotNull) match = !nulls_[i];
            mask[i] = match;
        }
    }
    
    void FilterNulls(bool* mask) const {
        for (size_t i = 0; i < size_; ++i) mask[i] = !nulls_[i];
    }
    
    void FilterNotNulls(bool* mask) const {
        for (size_t i = 0; i < size_; ++i) mask[i] = nulls_[i];
    }
    
private:
    std::vector<char> data_;
    std::vector<uint8_t> nulls_;
    size_t size_ = 0;
};

class ColumnBase {
public:
    virtual ~ColumnBase() = default;
    virtual DataType Type() const = 0;
    virtual size_t Size() const = 0;
    virtual size_t NullCount() const = 0;
    virtual void AppendNull() = 0;
    virtual void Reserve(size_t n) = 0;
    virtual std::unique_ptr<ColumnBase> Clone() const = 0;
    virtual void Filter(const bool* mask, size_t n, ColumnBase* out) const = 0;
    virtual void CopyTo(ColumnBase* dest, size_t offset, size_t count) const = 0;
    virtual std::unique_ptr<encoding::Encoder> CreateEncoder(EncodingType type) const = 0;
    virtual void Encode(encoding::Encoder* encoder, encoding::Buffer& out) const = 0;
    virtual void Decode(encoding::Decoder* decoder, const void* data, size_t size) = 0;

    // Read `count` fixed-size values starting at `offset` into `out`.
    // Values are stored contiguously (element size = TypeSize(Type())).
    virtual void ReadValues(size_t offset, size_t count, void* out) const = 0;
};

template<DataType T>
class TypedColumn : public ColumnBase {
    using Native = NativeType<T>;
    using Vec = ColumnVector<T>;
    
public:
    TypedColumn() = default;
    explicit TypedColumn(size_t capacity) : data_(capacity) {}
    
    DataType Type() const override { return T; }
    size_t Size() const override { return data_.Size(); }
    size_t NullCount() const override {
        size_t cnt = 0;
        for (size_t i = 0; i < data_.Size(); ++i) if (data_.IsNull(i)) ++cnt;
        return cnt;
    }
    
    void AppendNull() override { data_.AppendNulls(1); }
    void Reserve(size_t n) override { data_.Reserve(n); }
    
    void Append(const Native& value) { data_.Append(value); }
    void Append(const Native* values, size_t count) { data_.Append(values, nullptr, count); }
    void Append(const Native* values, const uint8_t* nulls, size_t count) { data_.Append(values, nulls, count); }
    
    Native& operator[](size_t i) { return data_[i]; }
    const Native& operator[](size_t i) const { return data_[i]; }
    bool IsNull(size_t i) const { return data_.IsNull(i); }
    void SetNull(size_t i, bool null = true) { data_.SetNull(i, null); }
    
    Native* Data() { return data_.Data(); }
    const Native* Data() const { return data_.Data(); }
    uint8_t* Nulls() { return data_.Nulls(); }
    const uint8_t* Nulls() const { return data_.Nulls(); }
    
    std::unique_ptr<ColumnBase> Clone() const override {
        auto copy = std::make_unique<TypedColumn<T>>();
        copy->data_ = data_;
        return copy;
    }
    
    void Filter(const bool* mask, size_t n, ColumnBase* out) const override {
        auto* out_col = static_cast<TypedColumn<T>*>(out);
        out_col->data_.Reserve(n);
        for (size_t i = 0; i < n; ++i) {
            if (mask[i]) {
                out_col->data_.Append(data_[i], data_.IsNull(i));
            }
        }
    }
    
    void CopyTo(ColumnBase* dest, size_t offset, size_t count) const override {
        auto* d = static_cast<TypedColumn<T>*>(dest);
        d->data_.Reserve(offset + count);
        d->data_.Append(Data() + offset, Nulls() + offset, count);
    }
    
    std::unique_ptr<encoding::Encoder> CreateEncoder(EncodingType type) const override {
        switch (type) {
            case EncodingType::Plain: return encoding::CreateEncoder<EncodingType::Plain, T>();
            case EncodingType::Dictionary: return encoding::CreateEncoder<EncodingType::Dictionary, T>();
            case EncodingType::RLE: return encoding::CreateEncoder<EncodingType::RLE, T>();
            case EncodingType::BitPack: return encoding::CreateEncoder<EncodingType::BitPack, T>();
            case EncodingType::FOR: return encoding::CreateEncoder<EncodingType::FOR, T>();
            case EncodingType::Delta: return encoding::CreateEncoder<EncodingType::Delta, T>();
            default: return nullptr;
        }
    }
    
    void Encode(encoding::Encoder* encoder, encoding::Buffer& out) const override {
        encoder->Encode(data_.Data(), data_.Size(), out);
    }
    
    void Decode(encoding::Decoder* decoder, const void* data, size_t size) override {
        size_t count = size / sizeof(Native);
        data_.Resize(count);
        decoder->Decode(data, size, data_.Data(), count);
    }
    
    template<PredicateType Pred>
    void FilterVector(const Native* value, bool* mask) const {
        data_.template Filter<Pred>(value, mask);
    }
    
    void ReadValues(size_t offset, size_t count, void* out) const override {
        std::memcpy(out, data_.Data() + offset, count * sizeof(Native));
    }
    
    void Sum(Native& result) const {
        result = VectorOps<T>::Sum(Data(), Size());
    }
    
    void Min(Native& result) const {
        result = VectorOps<T>::Min(Data(), Size());
    }
    
    void Max(Native& result) const {
        result = VectorOps<T>::Max(Data(), Size());
    }
    
private:
    Vec data_;
};

class ColumnFactory {
public:
    static std::unique_ptr<ColumnBase> Create(DataType type, size_t capacity = 0) {
        switch (type) {
            case DataType::Int32: return std::make_unique<TypedColumn<DataType::Int32>>(capacity);
            case DataType::Int64: return std::make_unique<TypedColumn<DataType::Int64>>(capacity);
            case DataType::Float: return std::make_unique<TypedColumn<DataType::Float>>(capacity);
            case DataType::Double: return std::make_unique<TypedColumn<DataType::Double>>(capacity);
            case DataType::Varchar: return std::make_unique<TypedColumn<DataType::Varchar>>(capacity);
            default: return nullptr;
        }
    }
};

} // namespace column
} // namespace columnar