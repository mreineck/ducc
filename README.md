DUCC 0.1
========

This is a collection of basic programming tools which can be handy in many
situations.


Installation
------------

### Requirements

- [Python 3](https://www.python.org/)
- [pybind11](https://github.com/pybind/pybind11)
- a C++17-capable C++ compiler (e.g. g++ from version 7 on or a recent clang++)

### Sources

The current version of DUCC can be obtained by cloning the repository via

    git clone https://gitlab.mpcdf.mpg.de/mtr/ducc.git

### Installation

In the following, we assume a Debian-based distribution. For other
distributions, the "apt" lines will need slight changes.

DUCC and its mandatory dependencies can be installed via:

    sudo apt-get install git python3 python3-pip python3-dev python3-pybind11 pybind11-dev
    pip3 install --user git+https://gitlab.mpcdf.mpg.de/mtr/ducc.git


DUCC components
===============

ducc.fft
--------

This package provides Fast Fourier, trigonometric and Hartley transforms with a
simple Python interface.

The central algorithms are derived from Paul Swarztrauber's FFTPACK code
(http://www.netlib.org/fftpack).

Features
- supports fully complex and half-complex (i.e. complex-to-real and
  real-to-complex) FFTs, discrete sine/cosine transforms and Hartley transforms
- achieves very high accuracy for all transforms
- supports multidimensional arrays and selection of the axes to be transformed
- supports single, double, and long double precision
- makes use of CPU vector instructions when performing 2D and higher-dimensional
  transforms
- supports prime-length transforms without degrading to O(N**2) performance
- has optional multi-threading support for multidimensional transforms


ducc.sht
--------

This package provides efficient spherical harmonic trasforms (SHTs). Its code
is derived from `libsharp`.


ducc.healpix
------------

This library provides Python bindings for the most important
functionality in Healpix C++. The design goals are
- similarity to the C++ interface (while respecting some Python peculiarities)
- simplicity (no optional function parameters)
- low function calling overhead


ducc.totalconvolve
------------------

Library for high-accuracy 4pi convolution on the sphere, which generates a
total convolution data cube from a set of sky and beam `a_lm` and computes
interpolated values for a given list of detector pointings.

Algorithmic details:
- the code uses `ducc.sht` SHTs to compute the data cube
- shared-memory parallelization is provided via standard C++ threads.
- for interpolation, the algorithm and kernel described in
  https://arxiv.org/abs/1808.06736 are used. This allows very efficient
  interpolation with user-adjustable accuracy.


ducc.wgridder
-------------

Library for high-accuracy gridding/degridding of radio interferometry datasets

Programming aspects
- written in C++11, fully portable
- shared-memory parallelization via and C++ threads.
- Python interface available
- kernel computation is performed on the fly, avoiding inaccuracies
  due to table lookup and reducing overall memory bandwidth

Numerical aspects
- uses the analytical gridding kernel presented in
  https://arxiv.org/abs/1808.06736
- uses the "improved W-stacking method" described in
  https://www.repository.cam.ac.uk/handle/1810/292298 (p. 139ff)
- in combination these two aspects allow extremely accurate gridding/degridding
  operations (L2 error compared to explicit DFTs can go below 1e-12) with
  reasonable resource consumption
