#include <gtest/gtest.h>
#include <columnar/predicate.h>
#include <columnar/types.h>
#include <vector>

using namespace columnar;
using namespace columnar::predicate;

TEST(PredicateTest, SimpleEqual) {
    auto p = PredicateBuilder().Eq<DataType::Int32>(0, 42).Build();
    
    EXPECT_NE(p, nullptr);
    EXPECT_EQ(p->Program().Instructions().size(), 3);
    EXPECT_EQ(p->Program().Instructions()[0].op, OpCode::LoadColumn);
    EXPECT_EQ(p->Program().Instructions()[1].op, OpCode::LoadConst);
    EXPECT_EQ(p->Program().Instructions()[2].op, OpCode::CompareEq);
}

TEST(PredicateTest, SimpleLessThan) {
    auto p = PredicateBuilder().Lt<DataType::Int64>(1, 100).Build();
    
    EXPECT_NE(p, nullptr);
    auto& insts = p->Program().Instructions();
    EXPECT_EQ(insts[2].op, OpCode::CompareLt);
}

TEST(PredicateTest, SimpleGreaterThan) {
    auto p = PredicateBuilder().Gt<DataType::Int32>(2, 3).Build();
    
    EXPECT_NE(p, nullptr);
    auto& insts = p->Program().Instructions();
    EXPECT_EQ(insts[2].op, OpCode::CompareGt);
}

TEST(PredicateTest, AndOrNot) {
    auto p = PredicateBuilder()
        .Eq<DataType::Int32>(0, 1)
        .And()
        .Eq<DataType::Int32>(1, 2)
        .Or()
        .Eq<DataType::Int32>(2, 3)
        .Not()
        .Build();
    
    EXPECT_NE(p, nullptr);
    auto& insts = p->Program().Instructions();
    
    int and_count = 0, or_count = 0, not_count = 0;
    for (const auto& inst : insts) {
        if (inst.op == OpCode::And) ++and_count;
        else if (inst.op == OpCode::Or) ++or_count;
        else if (inst.op == OpCode::Not) ++not_count;
    }
    
    EXPECT_EQ(and_count, 1);
    EXPECT_EQ(or_count, 1);
    EXPECT_EQ(not_count, 1);
}

TEST(PredicateTest, IsNull) {
    auto p = PredicateBuilder().Null(0).Build();
    
    EXPECT_NE(p, nullptr);
    auto& insts = p->Program().Instructions();
    EXPECT_EQ(insts[1].op, OpCode::IsNull);
}

TEST(PredicateTest, IsNotNull) {
    auto p = PredicateBuilder().NotNull(1).Build();
    
    EXPECT_NE(p, nullptr);
    auto& insts = p->Program().Instructions();
    EXPECT_EQ(insts[1].op, OpCode::IsNotNull);
}

TEST(PredicateTest, Constants) {
    auto p = PredicateBuilder()
        .Eq<DataType::Int32>(0, 100)
        .Build();
    
    EXPECT_NE(p, nullptr);
    EXPECT_GE(p->Program().Constants().size(), 1);
}

TEST(PredicateTest, MultipleColumns) {
    auto p = PredicateBuilder()
        .Eq<DataType::Int32>(0, 1)
        .Eq<DataType::Int32>(1, 2)
        .Build();
    
    EXPECT_NE(p, nullptr);
    EXPECT_GE(p->Program().ColumnIndices().size(), 1);
}

TEST(PredicateTest, ComplexExpression) {
    auto p = PredicateBuilder()
        .Gt<DataType::Int32>(0, 10)
        .And()
        .Lt<DataType::Int32>(0, 100)
        .And()
        .NotNull(1)
        .Build();
    
    EXPECT_NE(p, nullptr);
    
    int compare_count = 0;
    int logic_count = 0;
    for (const auto& inst : p->Program().Instructions()) {
        if (inst.op >= OpCode::CompareEq && inst.op <= OpCode::IsNotNull) ++compare_count;
        else if (inst.op == OpCode::And || inst.op == OpCode::Or || inst.op == OpCode::Not) ++logic_count;
    }
    
    EXPECT_EQ(compare_count, 3);
    EXPECT_EQ(logic_count, 2);
}

TEST(PredicateTest, SimdEvaluatorInt32) {
    std::vector<int32_t> data(100);
    for (size_t i = 0; i < 100; ++i) data[i] = static_cast<int32_t>(i);
    
    bool mask[100];
    int32_t value = 50;
    
    SimdPredicateEvaluator::EvaluateVector<PredicateType::Equal, DataType::Int32>(
        data.data(), &value, mask, 100);
    
    int count = 0;
    for (size_t i = 0; i < 100; ++i) if (mask[i]) ++count;
    EXPECT_EQ(count, 1);
    
    SimdPredicateEvaluator::EvaluateVector<PredicateType::LessThan, DataType::Int32>(
        data.data(), &value, mask, 100);
    
    count = 0;
    for (size_t i = 0; i < 100; ++i) if (mask[i]) ++count;
    EXPECT_EQ(count, 50);
}

TEST(PredicateTest, SimdEvaluatorInt64) {
    std::vector<int64_t> data(100);
    for (size_t i = 0; i < 100; ++i) data[i] = static_cast<int64_t>(i) * 1000;
    
    bool mask[100];
    int64_t value = 50000;
    
    SimdPredicateEvaluator::EvaluateVector<PredicateType::Equal, DataType::Int64>(
        data.data(), &value, mask, 100);
    
    int count = 0;
    for (size_t i = 0; i < 100; ++i) if (mask[i]) ++count;
    EXPECT_EQ(count, 1);
}

TEST(PredicateTest, LogicalOps) {
    bool mask1[10] = {true, false, true, false, true, false, true, false, true, false};
    bool mask2[10] = {false, true, false, true, false, true, false, true, false, true};
    bool out[10];
    
    SimdPredicateEvaluator::EvaluateAnd<DataType::Int32>(mask1, mask2, out, 10);
    for (int i = 0; i < 10; ++i) EXPECT_FALSE(out[i]);
    
    mask1[0] = true; mask2[0] = true;
    SimdPredicateEvaluator::EvaluateAnd<DataType::Int32>(mask1, mask2, out, 10);
    EXPECT_TRUE(out[0]);
    for (int i = 1; i < 10; ++i) EXPECT_FALSE(out[i]);
    
    SimdPredicateEvaluator::EvaluateOr<DataType::Int32>(mask1, mask2, out, 10);
    for (int i = 0; i < 10; ++i) EXPECT_TRUE(out[i]);
    
    SimdPredicateEvaluator::EvaluateNot<DataType::Int32>(out, out, 10);
    for (int i = 0; i < 10; ++i) EXPECT_FALSE(out[i]);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
