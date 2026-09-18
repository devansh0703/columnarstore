cmake_minimum_required(VERSION 3.20)

add_executable(bm_encoding
    bm_encoding.cpp
)

target_link_libraries(bm_encoding PRIVATE
    columnarstore
    benchmark::benchmark
)

target_compile_options(bm_encoding PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma>
    $<$<BOOL:${ENABLE_AVX512}>:-mavx512f -mavx512bw -mavx512vl>
)

add_executable(bm_scan
    bm_scan.cpp
)

target_link_libraries(bm_scan PRIVATE
    columnarstore
    benchmark::benchmark
)

target_compile_options(bm_scan PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma>
    $<$<BOOL:${ENABLE_AVX512}>:-mavx512f -mavx512bw -mavx512vl>
)

add_executable(bm_filter
    bm_filter.cpp
)

target_link_libraries(bm_filter PRIVATE
    columnarstore
    benchmark::benchmark
)

target_compile_options(bm_filter PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma>
    $<$<BOOL:${ENABLE_AVX512}>:-mavx512f -mavx512bw -mavx512vl>
)

add_executable(bm_aggregate
    bm_aggregate.cpp
)

target_link_libraries(bm_aggregate PRIVATE
    columnarstore
    benchmark::benchmark
)

target_compile_options(bm_aggregate PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma>
    $<$<BOOL:${ENABLE_AVX512}>:-mavx512f -mavx512bw -mavx512vl>
)

add_executable(bm_compare
    bm_compare.cpp
)

target_link_libraries(bm_compare PRIVATE
    columnarstore
    benchmark::benchmark
)

target_compile_options(bm_compare PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma>
    $<$<BOOL:${ENABLE_AVX512}>:-mavx512f -mavx512bw -mavx512vl>
)