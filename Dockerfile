FROM debian:testing-slim
RUN apt-get update && apt-get install -y \
      # General environment
      git python3-pip python3-pytest \
      # Ducc dependencies
      python3-scipy python3-pybind11 pybind11-dev \
      # Finufft dependencies
      libfftw3-dev python3-dotenv \
      # Demo dependency
      python3-numba \
      # Clean up
      && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/flatironinstitute/finufft \
      && cd finufft \
      && make -j 4 python
