FROM debian:testing-slim
RUN apt-get update && apt-get install -qq \
      # General environment
      git python3-pip python3-pytest gnupg wget \
      # Add latest llvm repository
      && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor > /etc/apt/trusted.gpg.d/llvm.gpg \
      && echo "deb http://apt.llvm.org/unstable/ llvm-toolchain-11 main\ndeb-src http://apt.llvm.org/unstable/ llvm-toolchain-11 main" | tee /etc/apt/sources.list.d/llvm.list \
      && apt-get update \
      && apt-get install -qq \
      # Ducc dependencies
      python3-scipy python3-pybind11 pybind11-dev python3-mpi4py mpi-default-dev mpi-default-bin \
      # Clang
      clang-11 \
      # Finufft dependencies
      libfftw3-dev python3-dotenv \
      # Demo dependency
      python3-numba \
      # Clean up
      && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/flatironinstitute/finufft \
      && cd finufft \
      && make -j 4 python
