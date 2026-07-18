#include <gtest/gtest.h>
#include <columnar/delta_store.h>
#include <vector>
#include <atomic>
#include <thread>

using namespace columnar::delta;

TEST(DeltaStoreTest, InsertAndGet) {
    DeltaStore store(2, 1000);
    
    std::vector<std::vector<uint8_t>> data(2);
    data[0] = {0x01, 0x00, 0x00, 0x00};
    data[1] = {0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    
    store.Insert(1, data, 100);
    store.Insert(2, data, 101);
    store.Insert(3, data, 102);
    
    EXPECT_EQ(store.PendingRows(), 3);
    EXPECT_FALSE(store.ShouldFlush());
    
    std::vector<uint8_t> out_data;
    uint64_t ts;
    
    bool found = store.Get(0, 1, out_data, ts);
    EXPECT_TRUE(found);
    EXPECT_EQ(out_data.size(), 4);
    EXPECT_EQ(out_data[0], 0x01);
    EXPECT_EQ(ts, 100);
    
    found = store.Get(0, 2, out_data, ts);
    EXPECT_TRUE(found);
    EXPECT_EQ(ts, 101);
    
    found = store.Get(0, 3, out_data, ts);
    EXPECT_TRUE(found);
    EXPECT_EQ(ts, 102);
}

TEST(DeltaStoreTest, Delete) {
    DeltaStore store(2, 1000);
    
    std::vector<std::vector<uint8_t>> data(2);
    data[0] = {0x01, 0x00, 0x00, 0x00};
    data[1] = {0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    
    store.Insert(1, data, 100);
    store.Insert(2, data, 101);
    store.Delete(1, 102);
    
    EXPECT_EQ(store.PendingRows(), 2);
    
    std::vector<uint8_t> out_data;
    uint64_t ts;
    
    bool found = store.Get(0, 1, out_data, ts);
    EXPECT_FALSE(found);
    
    found = store.Get(0, 2, out_data, ts);
    EXPECT_TRUE(found);
    EXPECT_EQ(ts, 101);
}

TEST(DeltaStoreTest, GetRange) {
    DeltaStore store(2, 1000);
    
    std::vector<std::vector<uint8_t>> data(2);
    data[0] = {0x01, 0x00, 0x00, 0x00};
    data[1] = {0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    
    for (uint64_t i = 1; i <= 100; ++i) {
        store.Insert(i, data, 100 + i);
    }
    
    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> results;
    store.GetRange(0, 50, 75, results);
    EXPECT_EQ(results.size(), 25);
    EXPECT_EQ(results[0].first, 50);
    EXPECT_EQ(results[24].first, 74);
    
    results.clear();
    store.GetRange(0, 1, 101, results);
    EXPECT_EQ(results.size(), 100);
}

TEST(DeltaStoreTest, Flush) {
    DeltaStore store(1, 10);
    
    std::vector<std::vector<uint8_t>> data(1);
    data[0] = {0x01, 0x00, 0x00, 0x00};
    
    for (int i = 0; i < 15; ++i) {
        store.Insert(i, data, 100 + i);
    }
    
    EXPECT_TRUE(store.ShouldFlush());
    EXPECT_EQ(store.PendingRows(), 15);
    
    store.BeginFlush();
    store.EndFlush(10);
    
    EXPECT_FALSE(store.IsFlushing());
    EXPECT_EQ(store.PendingRows(), 5);
}

TEST(DeltaStoreTest, DeltaStoreManager) {
    DeltaStoreManager manager(2, 5);
    
    std::vector<std::vector<uint8_t>> data(2);
    data[0] = {0x01, 0x00, 0x00, 0x00};
    data[1] = {0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    
    for (int i = 0; i < 10; ++i) {
        manager.CurrentStore()->Insert(i, data, 100 + i);
    }
    
    auto flush_store = manager.GetStoreForFlush();
    EXPECT_NE(flush_store, nullptr);
    EXPECT_EQ(flush_store->PendingRows(), 10);
    
    manager.FinishFlush(0, 10);
    EXPECT_EQ(flush_store->PendingRows(), 0);
}

TEST(DeltaStoreTest, ConcurrentInserts) {
    DeltaStore store(2, 100000);
    
    const int num_threads = 8;
    const int inserts_per_thread = 10000;
    std::atomic<int> total_inserted{0};
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&store, t, inserts_per_thread, &total_inserted]() {
            std::vector<std::vector<uint8_t>> data(2);
            data[0] = {static_cast<uint8_t>(t), 0x00, 0x00, 0x00};
            data[1] = {static_cast<uint8_t>(t), 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
            
            for (int i = 0; i < inserts_per_thread; ++i) {
                uint64_t row_id = static_cast<uint64_t>(t) * inserts_per_thread + i + 1;
                store.Insert(row_id, data, row_id);
                total_inserted.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    
    for (auto& th : threads) th.join();
    
    EXPECT_EQ(store.PendingRows(), num_threads * inserts_per_thread);
    EXPECT_EQ(total_inserted.load(), num_threads * inserts_per_thread);
}

TEST(DeltaStoreTest, MemoryUsage) {
    DeltaStore store(2, 10000);
    
    std::vector<std::vector<uint8_t>> data(2);
    data[0].assign(100, 0xFF);
    data[1].assign(200, 0xFF);
    
    for (int i = 0; i < 100; ++i) {
        store.Insert(i, data, 100 + i);
    }
    
    size_t usage = store.MemoryUsage();
    EXPECT_GT(usage, 0);
    EXPECT_LT(usage, 100 * (100 + 200) * 2);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}