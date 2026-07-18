FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    libzstd-dev \
    liblz4-dev \
    libxxhash-dev \
    libgtest-dev \
    libbenchmark-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/columnarstore

COPY . .

RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DBUILD_BENCHMARKS=ON \
    -DENABLE_AVX512=ON \
    && cmake --build build --parallel $(nproc)

RUN ctest --test-dir build --output-on-failure

CMD ["/bin/bash"]
