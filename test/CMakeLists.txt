cmake_minimum_required(VERSION 3.20)

add_executable(test_encoding
    test_encoding.cpp
)

target_link_libraries(test_encoding PRIVATE
    columnarstore
    GTest::gtest
    GTest::gtest_main
)

target_compile_options(test_encoding PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma>
)

add_executable(test_segment
    test_segment.cpp
)

target_link_libraries(test_segment PRIVATE
    columnarstore
    GTest::gtest
    GTest::gtest_main
)

target_compile_options(test_segment PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma>
)

add_executable(test_delta_store
    test_delta_store.cpp
)

target_link_libraries(test_delta_store PRIVATE
    columnarstore
    GTest::gtest
    GTest::gtest_main
)

target_compile_options(test_delta_store PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma>
)

add_executable(test_merger
    test_merger.cpp
)

target_link_libraries(test_merger PRIVATE
    columnarstore
    GTest::gtest
    GTest::gtest_main
)

target_compile_options(test_merger PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma>
)

add_executable(test_scanner
    test_scanner.cpp
)

target_link_libraries(test_scanner PRIVATE
    columnarstore
    GTest::gtest
    GTest::gtest_main
)

target_compile_options(test_scanner PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma>
)

add_executable(test_predicate
    test_predicate.cpp
)

target_link_libraries(test_predicate PRIVATE
    columnarstore
    GTest::gtest
    GTest::gtest_main
)

target_compile_options(test_predicate PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma>
)
enable_testing()
add_test(NAME test_encoding COMMAND test_encoding)
add_test(NAME test_segment COMMAND test_segment)
add_test(NAME test_delta_store COMMAND test_delta_store)
add_test(NAME test_merger COMMAND test_merger)
add_test(NAME test_scanner COMMAND test_scanner)
add_test(NAME test_predicate COMMAND test_predicate)
