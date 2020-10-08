FROM debian:buster-slim
RUN apt-get update && apt-get install -y git python3-pip python3-scipy python3-pytest python3-pybind11 pybind11-dev clang && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -qq libfftw3-dev python3-dotenv && rm -rf /var/lib/apt/lists/* && git clone --depth 1 https://github.com/flatironinstitute/finufft && cd finufft && make -j 4 python
