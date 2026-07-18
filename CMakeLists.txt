cmake_minimum_required(VERSION 3.20)
project(ColumnarStore VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-error=abi -O3 -fext-numeric-literals -std=c++20")
add_compile_options(-Wno-error=abi)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fsanitize=address,undefined")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")

option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(ENABLE_AVX512 "Enable AVX-512 optimizations" ON)

find_package(Threads REQUIRED)

if(BUILD_TESTS)
    find_package(GTest REQUIRED)
endif()

if(BUILD_BENCHMARKS)
    find_package(benchmark REQUIRED)
endif()

find_library(ZSTD_LIBRARY NAMES zstd PATHS /usr/lib/x86_64-linux-gnu /usr/lib)
find_path(ZSTD_INCLUDE_DIR NAMES zstd.h PATHS /usr/include /usr/local/include)

find_library(LZ4_LIBRARY NAMES lz4 PATHS /usr/lib/x86_64-linux-gnu /usr/lib)
find_path(LZ4_INCLUDE_DIR NAMES lz4.h lz4hc.h PATHS /usr/include /usr/local/include)

if(ZSTD_LIBRARY AND ZSTD_INCLUDE_DIR)
    message(STATUS "Found ZSTD: ${ZSTD_LIBRARY}")
    add_library(ZSTD::ZSTD UNKNOWN IMPORTED)
    set_target_properties(ZSTD::ZSTD PROPERTIES
        IMPORTED_LOCATION "${ZSTD_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ZSTD_INCLUDE_DIR}"
    )
else()
    message(FATAL_ERROR "ZSTD not found")
endif()

if(LZ4_LIBRARY AND LZ4_INCLUDE_DIR)
    message(STATUS "Found LZ4: ${LZ4_LIBRARY}")
    add_library(LZ4::LZ4 UNKNOWN IMPORTED)
    set_target_properties(LZ4::LZ4 PROPERTIES
        IMPORTED_LOCATION "${LZ4_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${LZ4_INCLUDE_DIR}"
    )
else()
    message(FATAL_ERROR "LZ4 not found")
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" HAVE_AVX2)
check_cxx_compiler_flag("-mavx512f" HAVE_AVX512F)
check_cxx_compiler_flag("-mavx512bw" HAVE_AVX512BW)
check_cxx_compiler_flag("-mavx512vl" HAVE_AVX512VL)

add_library(columnarstore
    src/segment.cpp
    src/dictionary.cpp
    src/rle.cpp
    src/bitpack.cpp
    src/for.cpp
    src/encoding/zstd.cpp
    src/encoding/instantiations.cpp
    src/zone_map.cpp
    src/bloom_filter.cpp
    src/delta_store.cpp
    src/merger.cpp
    src/scanner.cpp
    src/predicate.cpp
)

target_include_directories(columnarstore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(columnarstore PUBLIC
    ZSTD::ZSTD
    LZ4::LZ4
    Threads::Threads
)

target_compile_options(columnarstore PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma>
    $<$<BOOL:${ENABLE_AVX512}>:-mavx512f -mavx512bw -mavx512vl>
)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

install(TARGETS columnarstore
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/columnar/ DESTINATION include/columnar)
install(DIRECTORY include/simd/ DESTINATION include/simd)