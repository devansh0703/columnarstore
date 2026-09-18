cmake_minimum_required(VERSION 3.20)

# Read version from VERSION (single source of truth, also used by CI/release tooling)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" COLUMNARSTORE_VERSION)
string(STRIP "${COLUMNARSTORE_VERSION}" COLUMNARSTORE_VERSION)

project(ColumnarStore VERSION ${COLUMNARSTORE_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-error=abi -O3 -fext-numeric-literals -std=c++20")
add_compile_options(-Wno-error=abi)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fsanitize=address,undefined")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")

option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(COLUMNARSTORE_BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(ENABLE_AVX512 "Enable AVX-512 optimizations" ON)

include(GNUInstallDirs)
include(FetchContent)

find_package(Threads REQUIRED)

if(BUILD_TESTS)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        message(STATUS "GTest not found — fetching via FetchContent")
        FetchContent_Declare(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.15.2
        )
        # Prevent gtest from overriding our compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
endif()

if(BUILD_BENCHMARKS)
    find_package(benchmark QUIET)
    if(NOT benchmark_FOUND)
        message(STATUS "google/benchmark not found — fetching via FetchContent")
        FetchContent_Declare(benchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v1.9.1
        )
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(benchmark)
    endif()
endif()

# --- Required compression libraries (with FetchContent fallback) -------------
# NOTE: no find_package(zstd) here — some systems expose a zstd CMake package
# that does not define the ZSTD::ZSTD target we link against.
find_library(ZSTD_LIBRARY NAMES zstd PATHS /usr/lib/x86_64-linux-gnu /usr/lib)
find_path(ZSTD_INCLUDE_DIR NAMES zstd.h PATHS /usr/include /usr/local/include)
if(ZSTD_LIBRARY AND ZSTD_INCLUDE_DIR)
    message(STATUS "Found ZSTD: ${ZSTD_LIBRARY}")
    add_library(ZSTD::ZSTD UNKNOWN IMPORTED)
    set_target_properties(ZSTD::ZSTD PROPERTIES
        IMPORTED_LOCATION "${ZSTD_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ZSTD_INCLUDE_DIR}"
    )
    set(COLUMNARSTORE_ZSTD_TARGET ZSTD::ZSTD)
else()
    message(STATUS "libzstd not found — fetching via FetchContent")
    FetchContent_Declare(zstd
        GIT_REPOSITORY https://github.com/facebook/zstd.git
        GIT_TAG v1.5.6
        SOURCE_SUBDIR build/cmake
    )
    set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(zstd)
    set(COLUMNARSTORE_ZSTD_TARGET libzstd_static)
endif()

find_library(LZ4_LIBRARY NAMES lz4 PATHS /usr/lib/x86_64-linux-gnu /usr/lib)
find_path(LZ4_INCLUDE_DIR NAMES lz4.h lz4hc.h PATHS /usr/include /usr/local/include)
if(LZ4_LIBRARY AND LZ4_INCLUDE_DIR)
    message(STATUS "Found LZ4: ${LZ4_LIBRARY}")
    add_library(LZ4::LZ4 UNKNOWN IMPORTED)
    set_target_properties(LZ4::LZ4 PROPERTIES
        IMPORTED_LOCATION "${LZ4_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${LZ4_INCLUDE_DIR}"
    )
    set(COLUMNARSTORE_LZ4_TARGET LZ4::LZ4)
else()
    message(STATUS "liblz4 not found — fetching via FetchContent")
    FetchContent_Declare(lz4
        GIT_REPOSITORY https://github.com/lz4/lz4.git
        GIT_TAG v1.9.4
        SOURCE_SUBDIR build/cmake
    )
    set(LZ4_BUILD_CLI OFF CACHE BOOL "" FORCE)
    set(LZ4_BUILD_LIBS ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(lz4)
    set(COLUMNARSTORE_LZ4_TARGET LZ4::LZ4_static)
endif()

# --- Core library ------------------------------------------------------------
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

add_library(columnarstore::columnarstore ALIAS columnarstore)

target_include_directories(columnarstore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(columnarstore PUBLIC
    ${COLUMNARSTORE_ZSTD_TARGET}
    ${COLUMNARSTORE_LZ4_TARGET}
    Threads::Threads
)

# Warnings apply to our code only, not to consumers of the installed target.
target_compile_options(columnarstore PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wno-error=abi>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx2 -mfma>
    $<$<BOOL:${ENABLE_AVX512}>:-mavx512f -mavx512bw -mavx512vl>
)

set_target_properties(columnarstore PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# --- Tests / benchmarks / examples -------------------------------------------
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

if(COLUMNARSTORE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# --- Install / export ----------------------------------------------------------
install(TARGETS columnarstore
    EXPORT columnarstoreTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/columnar/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/columnar)
install(DIRECTORY include/simd/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/simd)

set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)
export(PACKAGE columnarstore)

install(EXPORT columnarstoreTargets
    FILE columnarstoreTargets.cmake
    NAMESPACE columnarstore::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/columnarstore
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/columnarstoreConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/columnarstoreConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/columnarstoreConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/columnarstore
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/columnarstoreConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/columnarstoreConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/columnarstore
)

# pkg-config file (for Makefile-based consumers)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/columnarstore.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/columnarstore.pc"
    @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/columnarstore.pc"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# --- CPack (source + binary packages) ------------------------------------------
set(CPACK_PACKAGE_NAME "columnarstore")
set(CPACK_PACKAGE_VENDOR "ColumnarStore contributors")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "SIMD-accelerated columnar storage library for analytical workloads")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_SOURCE_IGNORE_FILES "/build/;/\\\\.git/;/\\\\.github/;~$")
include(CPack)
