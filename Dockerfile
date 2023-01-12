FROM debian:stable-slim
#FROM i386/debian:stable-slim
RUN apt-get update && apt-get install -qq \
      # General environment
      git python3-pip python3-pytest gnupg wget \
      && apt-get update \
      && apt-get install -qq \
      # Ducc dependencies
      python3-scipy python3-numpy python3-pybind11 pybind11-dev python3-mpi4py mpi-default-dev mpi-default-bin \
      # Clang
      clang \
      # Doxygen
      doxygen graphviz \
      && rm -rf /var/lib/apt/lists/*
# RUN pip install numba   # demo dependency
RUN pip install sphinx==5.3 pydata-sphinx-theme
