DUCC 0.1
========

This is a collection of basic programming tools which can be handy in many
situations.


Installation
------------

### Requirements

- [Python 3](https://www.python.org/)
- [pybind11](https://github.com/pybind/pybind11)
- a C++17-capable C++ compiler (tested with g++ version 7 or newer and clang++;
  recent versions of MSVC on Windows also work, but are tested less frequently)

### Sources

The latest version of DUCC can be obtained by cloning the repository via

    git clone https://gitlab.mpcdf.mpg.de/mtr/ducc.git

### Installation

In the following, we assume a Debian-based distribution. For other
distributions, the "apt" lines will need slight changes.

DUCC and its mandatory dependencies can be installed via:

    sudo apt-get install git python3 python3-pip python3-dev python3-pybind11 pybind11-dev
    pip3 install --user git+https://gitlab.mpcdf.mpg.de/mtr/ducc.git


Installing multiple versions simultaneously
===========================================

The interfaces of the DUCC components are expected to evolve over time; whenever
an interface changes in a manner that is not backwards compatible, the DUCC
version number will increase. As a consequence it might happen that one part of
a Python code may use an older version of DUCC while at the same time another
part requires a newer version. Since DUCC's version number is included in the
module name itself (the module is not called "ducc", but rather "ducc_x_y"),
this is not a problem, as multiple DUCC versions can be installed
simultaneously.
The latest patch levels of a given DUCC version will always be available at the
HEAD of the git branch with the respective name. In other words, if you need
the latest incarnation of DUCC 0.1, this will be in branch "ducc_0_1" of the
git repository, and it will be installed as the package "ducc_0_1".
Later versions (like ducc_0_2 or ducc_1_0) will be maintained on new branches
and will be installed as "ducc_0_2" and "ducc_1_0", so that there will be no
conflict with potentially installed older versions.


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
- the code uses `ducc.sht` SHTs and `ducc.fft` FFTs to compute the data cube
- shared-memory parallelization is provided via standard C++ threads.
- for interpolation, the algorithm and kernel described in
  https://arxiv.org/abs/1808.06736 are used. This allows very efficient
  interpolation with user-adjustable accuracy.


ducc.wgridder
-------------

Library for high-accuracy gridding/degridding of radio interferometry datasets

Programming aspects
- written in C++17, fully portable
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
