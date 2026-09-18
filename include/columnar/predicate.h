#pragma once

#include <columnar/types.h>
#include <columnar/zone_map.h>
#include <columnar/bloom_filter.h>
#include <columnar/column/column.h>
#include <vector>
#include <cstdint>
#include <memory>
#include <variant>
#include <string>
#include <functional>
#include <cstring>

namespace columnar {
namespace predicate {

enum class OpCode : uint8_t {
    Nop = 0,
    LoadConst = 1,
    LoadColumn = 2,
    CompareEq = 3,
    CompareNe = 4,
    CompareLt = 5,
    CompareLe = 6,
    CompareGt = 7,
    CompareGe = 8,
    IsNull = 9,
    IsNotNull = 10,
    And = 11,
    Or = 12,
    Not = 13,
    BranchIfFalse = 14,
    BranchIfTrue = 15,
    Return = 16,
    LoadMask = 17,
    AndMask = 18,
    OrMask = 19,
    NotMask = 20,
    PopCount = 21,
    SetResult = 22,
};

struct Instruction {
    OpCode op;
    uint32_t operand_a;
    uint32_t operand_b;
    uint32_t operand_c;
};

class BytecodeProgram {
    std::vector<Instruction> instructions_;
    std::vector<int64_t> constants_;
    std::vector<size_t> column_indices_;
    
public:
    size_t AddConstant(int64_t value) {
        constants_.push_back(value);
        return constants_.size() - 1;
    }
    
    size_t AddConstant(double value) {
        int64_t raw;
        std::memcpy(&raw, &value, sizeof(double));
        constants_.push_back(raw);
        return constants_.size() - 1;
    }
    
    size_t AddColumn(size_t col_idx) {
        column_indices_.push_back(col_idx);
        return column_indices_.size() - 1;
    }
    
    void Emit(OpCode op, uint32_t a = 0, uint32_t b = 0, uint32_t c = 0) {
        instructions_.push_back({op, a, b, c});
    }
    
    const std::vector<Instruction>& Instructions() const { return instructions_; }
    const std::vector<int64_t>& Constants() const { return constants_; }
    const std::vector<size_t>& ColumnIndices() const { return column_indices_; }
    
    size_t NumInstructions() const { return instructions_.size(); }
};

class Predicate {
    BytecodeProgram program_;
    std::vector<PredicateType> column_predicates_;
    std::vector<std::vector<uint8_t>> column_values_;
    
public:
    Predicate() = default;
    
    template<DataType T>
    Predicate& Equal(size_t column, typename TypeTraits<T>::Type value) {
        column_predicates_.push_back(PredicateType::Equal);
        size_t const_idx = program_.AddConstant(static_cast<int64_t>(value));
        size_t col_idx = program_.AddColumn(column);
        program_.Emit(OpCode::LoadColumn, col_idx);
        program_.Emit(OpCode::LoadConst, const_idx);
        program_.Emit(OpCode::CompareEq);
        return *this;
    }
    
    template<DataType T>
    Predicate& NotEqual(size_t column, typename TypeTraits<T>::Type value) {
        column_predicates_.push_back(PredicateType::NotEqual);
        size_t const_idx = program_.AddConstant(static_cast<int64_t>(value));
        size_t col_idx = program_.AddColumn(column);
        program_.Emit(OpCode::LoadColumn, col_idx);
        program_.Emit(OpCode::LoadConst, const_idx);
        program_.Emit(OpCode::CompareNe);
        return *this;
    }
    
    template<DataType T>
    Predicate& LessThan(size_t column, typename TypeTraits<T>::Type value) {
        column_predicates_.push_back(PredicateType::LessThan);
        size_t const_idx = program_.AddConstant(static_cast<int64_t>(value));
        size_t col_idx = program_.AddColumn(column);
        program_.Emit(OpCode::LoadColumn, col_idx);
        program_.Emit(OpCode::LoadConst, const_idx);
        program_.Emit(OpCode::CompareLt);
        return *this;
    }
    
    template<DataType T>
    Predicate& GreaterThan(size_t column, typename TypeTraits<T>::Type value) {
        column_predicates_.push_back(PredicateType::GreaterThan);
        size_t const_idx = program_.AddConstant(static_cast<int64_t>(value));
        size_t col_idx = program_.AddColumn(column);
        program_.Emit(OpCode::LoadColumn, col_idx);
        program_.Emit(OpCode::LoadConst, const_idx);
        program_.Emit(OpCode::CompareGt);
        return *this;
    }
    
    Predicate& IsNull(size_t column) {
        column_predicates_.push_back(PredicateType::IsNull);
        size_t col_idx = program_.AddColumn(column);
        program_.Emit(OpCode::LoadColumn, col_idx);
        program_.Emit(OpCode::IsNull);
        return *this;
    }
    
    Predicate& IsNotNull(size_t column) {
        column_predicates_.push_back(PredicateType::IsNotNull);
        size_t col_idx = program_.AddColumn(column);
        program_.Emit(OpCode::LoadColumn, col_idx);
        program_.Emit(OpCode::IsNotNull);
        return *this;
    }
    
    Predicate& And() {
        program_.Emit(OpCode::And);
        return *this;
    }
    
    Predicate& Or() {
        program_.Emit(OpCode::Or);
        return *this;
    }
    
    Predicate& Not() {
        program_.Emit(OpCode::Not);
        return *this;
    }
    
    const BytecodeProgram& Program() const { return program_; }
    const std::vector<PredicateType>& ColumnPredicates() const { return column_predicates_; }
    
    bool CanMatchZoneMap(const std::vector<std::shared_ptr<columnar::zone_map::ZoneMap>>& zone_maps) const {
        const auto& preds = column_predicates_;
        const auto& constants = program_.Constants();
        const auto& col_indices = program_.ColumnIndices();
        
        for (size_t i = 0; i < preds.size(); ++i) {
            if (i >= zone_maps.size() || !zone_maps[i]) continue;
            
            size_t col = i < col_indices.size() ? col_indices[i] : 0;
            if (col >= zone_maps.size() || !zone_maps[col]) continue;
            
            for (size_t b = 0; b < zone_maps[col]->NumBlocks(); ++b) {
                const auto& entry = zone_maps[col]->Block(b);
                if (entry.row_count == 0) continue;
                if (preds[i] == PredicateType::IsNull) continue;
                if (preds[i] == PredicateType::IsNotNull) {
                    if (entry.row_count > entry.null_count) continue;
                    return false;
                }
                if (!entry.has_min_max) continue;
                
                int64_t min_v = entry.min_int;
                int64_t max_v = entry.max_int;
                if (i < constants.size()) {
                    int64_t val = constants[i];
                    bool possible = true;
                    switch (preds[i]) {
                        case PredicateType::Equal: possible = (min_v <= val && val <= max_v); break;
                        case PredicateType::NotEqual: possible = !(min_v == val && max_v == val); break;
                        case PredicateType::LessThan: possible = (min_v < val); break;
                        case PredicateType::LessEqual: possible = (min_v <= val); break;
                        case PredicateType::GreaterThan: possible = (max_v > val); break;
                        case PredicateType::GreaterEqual: possible = (max_v >= val); break;
                        default: possible = true; break;
                    }
                    if (!possible) return false;
                }
            }
        }
        return true;
    }
    
    bool CanMatchBloom(const std::vector<std::shared_ptr<columnar::bloom_filter::BloomFilter>>& blooms) const {
        const auto& preds = column_predicates_;
        const auto& constants = program_.Constants();
        const auto& col_indices = program_.ColumnIndices();
        
        for (size_t i = 0; i < preds.size(); ++i) {
            if (preds[i] != PredicateType::Equal) continue;
            size_t col = i < col_indices.size() ? col_indices[i] : 0;
            if (col >= blooms.size() || !blooms[col]) continue;
            if (i >= constants.size()) continue;
            
            int64_t val = constants[i];
            if (!blooms[col]->MightContain(val)) return false;
        }
        return true;
    }
};

class BytecodeInterpreter {
    const BytecodeProgram& program_;
    std::vector<int64_t> stack_;
    // Deferred AND/OR for eager (infix-style) programs, where PredicateBuilder
    // emits the operator between its two operands instead of after them.
    bool pending_and_ = false;
    bool pending_or_ = false;
    
    void FoldPending() {
        if (pending_and_ && stack_.size() >= 2) {
            pending_and_ = false;
            int64_t b = stack_.back(); stack_.pop_back();
            int64_t a = stack_.back(); stack_.pop_back();
            stack_.push_back((a && b) ? 1 : 0);
        }
        if (pending_or_ && stack_.size() >= 2) {
            pending_or_ = false;
            int64_t b = stack_.back(); stack_.pop_back();
            int64_t a = stack_.back(); stack_.pop_back();
            stack_.push_back((a || b) ? 1 : 0);
        }
    }
    
public:
    BytecodeInterpreter(const BytecodeProgram& prog, size_t /*vector_width*/ = 16)
        : program_(prog) {}
    
    void Execute(const std::vector<const void*>& columns, 
                 const std::vector<const bool*>& /*nulls*/,
                 size_t count, bool* output) {
        for (size_t i = 0; i < count; ++i) {
            output[i] = EvaluateRow(columns, i);
        }
    }
    
    // Evaluate the program for a single row. The result is the final value left
    // on the stack, or the value pushed by an explicit Return instruction.
    bool EvaluateRow(const std::vector<const void*>& columns, size_t row) {
        stack_.clear();
        pending_and_ = false;
        pending_or_ = false;
        
        for (const auto& instr : program_.Instructions()) {
            switch (instr.op) {
                    case OpCode::LoadColumn: {
                        size_t col = program_.ColumnIndices()[instr.operand_a];
                        const int64_t* data = static_cast<const int64_t*>(columns[col]);
                        stack_.push_back(data[row]);
                        break;
                    }
                    case OpCode::LoadConst: {
                        int64_t val = program_.Constants()[instr.operand_a];
                        stack_.push_back(val);
                        break;
                    }
                    case OpCode::CompareEq: {
                        int64_t b = stack_.back(); stack_.pop_back();
                        int64_t a = stack_.back(); stack_.pop_back();
                        stack_.push_back(a == b ? 1 : 0);
                        FoldPending();
                        break;
                    }
                    case OpCode::CompareNe: {
                        int64_t b = stack_.back(); stack_.pop_back();
                        int64_t a = stack_.back(); stack_.pop_back();
                        stack_.push_back(a != b ? 1 : 0);
                        FoldPending();
                        break;
                    }
                    case OpCode::CompareLt: {
                        int64_t b = stack_.back(); stack_.pop_back();
                        int64_t a = stack_.back(); stack_.pop_back();
                        stack_.push_back(a < b ? 1 : 0);
                        FoldPending();
                        break;
                    }
                    case OpCode::CompareLe: {
                        int64_t b = stack_.back(); stack_.pop_back();
                        int64_t a = stack_.back(); stack_.pop_back();
                        stack_.push_back(a <= b ? 1 : 0);
                        FoldPending();
                        break;
                    }
                    case OpCode::CompareGt: {
                        int64_t b = stack_.back(); stack_.pop_back();
                        int64_t a = stack_.back(); stack_.pop_back();
                        stack_.push_back(a > b ? 1 : 0);
                        FoldPending();
                        break;
                    }
                    case OpCode::CompareGe: {
                        int64_t b = stack_.back(); stack_.pop_back();
                        int64_t a = stack_.back(); stack_.pop_back();
                        stack_.push_back(a >= b ? 1 : 0);
                        FoldPending();
                        break;
                    }
                    case OpCode::IsNull: {
                        int64_t a = stack_.back(); stack_.pop_back();
                        stack_.push_back(a == 0 ? 1 : 0);
                        break;
                    }
                    case OpCode::IsNotNull: {
                        int64_t a = stack_.back(); stack_.pop_back();
                        stack_.push_back(a != 0 ? 1 : 0);
                        break;
                    }
                    case OpCode::And: {
                        if (stack_.size() >= 2) {
                            // Postfix form: both operands already evaluated.
                            int64_t b = stack_.back(); stack_.pop_back();
                            int64_t a = stack_.back(); stack_.pop_back();
                            stack_.push_back((a && b) ? 1 : 0);
                        } else {
                            // Eager form: right operand comes later; defer.
                            pending_and_ = true;
                        }
                        break;
                    }
                    case OpCode::Or: {
                        if (stack_.size() >= 2) {
                            int64_t b = stack_.back(); stack_.pop_back();
                            int64_t a = stack_.back(); stack_.pop_back();
                            stack_.push_back((a || b) ? 1 : 0);
                        } else {
                            pending_or_ = true;
                        }
                        break;
                    }
                    case OpCode::Not: {
                        int64_t a = stack_.back(); stack_.pop_back();
                        stack_.push_back(a == 0 ? 1 : 0);
                        break;
                    }
                    case OpCode::Return: {
                        int64_t result = stack_.back(); stack_.pop_back();
                        return result != 0;
                    }
                    case OpCode::LoadMask: {
                        size_t col = program_.ColumnIndices()[instr.operand_a];
                        const int64_t* data = static_cast<const int64_t*>(columns[col]);
                        stack_.push_back(data[row]);
                        break;
                    }
                    case OpCode::AndMask: {
                        int64_t b = stack_.back(); stack_.pop_back();
                        int64_t a = stack_.back(); stack_.pop_back();
                        stack_.push_back(a & b);
                        break;
                    }
                    case OpCode::OrMask: {
                        int64_t b = stack_.back(); stack_.pop_back();
                        int64_t a = stack_.back(); stack_.pop_back();
                        stack_.push_back(a | b);
                        break;
                    }
                    case OpCode::BranchIfFalse: {
                        stack_.pop_back();
                        break;
                    }
                    default: break;
                }
            }
            return !stack_.empty() && stack_.back() != 0;
    }
};

class SimdPredicateEvaluator {
    std::vector<Predicate> predicates_;
    std::vector<size_t> columns_;
    
public:
    template<PredicateType Pred, DataType T>
    static void EvaluateVector(const typename TypeTraits<T>::Type* data,
                               const typename TypeTraits<T>::Type* value,
                               bool* mask, size_t count) {
        using VecTraits = column::VectorTraits<T>;
        using Vec = typename VecTraits::Vec;
        
        Vec vval = VecTraits::Set1(*value);
        size_t width = VecTraits::Width;
        
        for (size_t i = 0; i + width <= count; i += width) {
            Vec vdata = VecTraits::Load(data + i);
            auto m = VecTraits::CmpEq(vdata, vval);
            
            if constexpr (Pred == PredicateType::Equal) m = VecTraits::CmpEq(vdata, vval);
            else if constexpr (Pred == PredicateType::NotEqual) m = ~VecTraits::CmpEq(vdata, vval);
            else if constexpr (Pred == PredicateType::LessThan) m = VecTraits::CmpLt(vdata, vval);
            else if constexpr (Pred == PredicateType::LessEqual) m = VecTraits::CmpLe(vdata, vval);
            else if constexpr (Pred == PredicateType::GreaterThan) m = VecTraits::CmpLt(vval, vdata);
            else if constexpr (Pred == PredicateType::GreaterEqual) m = VecTraits::CmpLe(vval, vdata);
            else m = VecTraits::CmpEq(vdata, vval);
            
            for (size_t j = 0; j < width; ++j) {
                mask[i + j] = (m >> j) & 1;
            }
        }
        
        for (size_t i = count - count % width; i < count; ++i) {
            bool match = false;
            if constexpr (Pred == PredicateType::Equal) match = data[i] == *value;
            else if constexpr (Pred == PredicateType::NotEqual) match = data[i] != *value;
            else if constexpr (Pred == PredicateType::LessThan) match = data[i] < *value;
            else if constexpr (Pred == PredicateType::LessEqual) match = data[i] <= *value;
            else if constexpr (Pred == PredicateType::GreaterThan) match = data[i] > *value;
            else if constexpr (Pred == PredicateType::GreaterEqual) match = data[i] >= *value;
            mask[i] = match;
        }
    }
    
    template<DataType T>
    static void EvaluateAnd(const bool* mask1, const bool* mask2, bool* out, size_t count) {
        for (size_t i = 0; i < count; ++i) out[i] = mask1[i] && mask2[i];
    }
    
    template<DataType T>
    static void EvaluateOr(const bool* mask1, const bool* mask2, bool* out, size_t count) {
        for (size_t i = 0; i < count; ++i) out[i] = mask1[i] || mask2[i];
    }
    
    template<DataType T>
    static void EvaluateNot(const bool* mask, bool* out, size_t count) {
        for (size_t i = 0; i < count; ++i) out[i] = !mask[i];
    }
};

class PredicateBuilder {
    std::unique_ptr<Predicate> current_;
    
public:
    PredicateBuilder() : current_(std::make_unique<Predicate>()) {}
    
    template<DataType T>
    PredicateBuilder& Eq(size_t col, typename TypeTraits<T>::Type val) {
        current_->template Equal<T>(col, val);
        return *this;
    }
    
    template<DataType T>
    PredicateBuilder& Ne(size_t col, typename TypeTraits<T>::Type val) {
        current_->template NotEqual<T>(col, val);
        return *this;
    }
    
    template<DataType T>
    PredicateBuilder& Lt(size_t col, typename TypeTraits<T>::Type val) {
        current_->template LessThan<T>(col, val);
        return *this;
    }
    
    template<DataType T>
    PredicateBuilder& Gt(size_t col, typename TypeTraits<T>::Type val) {
        current_->template GreaterThan<T>(col, val);
        return *this;
    }
    
    PredicateBuilder& Null(size_t col) {
        current_->IsNull(col);
        return *this;
    }
    
    PredicateBuilder& NotNull(size_t col) {
        current_->IsNotNull(col);
        return *this;
    }
    
    PredicateBuilder& And() {
        current_->And();
        return *this;
    }
    
    PredicateBuilder& Or() {
        current_->Or();
        return *this;
    }
    
    PredicateBuilder& Not() {
        current_->Not();
        return *this;
    }
    
    std::unique_ptr<Predicate> Build() { return std::move(current_); }
};

inline std::unique_ptr<Predicate> MakePredicate() {
    return std::make_unique<Predicate>();
}

template<DataType T>
inline std::unique_ptr<Predicate> MakeEq(size_t col, typename TypeTraits<T>::Type val) {
    auto p = MakePredicate();
    p->template Equal<T>(col, val);
    return p;
}

} // namespace predicate
} // namespace columnar
