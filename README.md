Distinctly Useful Code Collection (DUCC)
========================================

This is a collection of basic programming tools for numerical computation,
including Fast Fourier Transforms, Spherical Harmonic Transforms, non-equispaced
Fourier transforms, as well as some concrete applications like 4pi convolution
on the sphere and gridding/degridding of radio interferometry data.

The code is written in C++17, but provides a simple and comprehensive Python
interface.

### Requirements

- [Python >= 3.6](https://www.python.org/)
- [pybind11](https://github.com/pybind/pybind11)
- a C++17-capable compiler (tested with g++ version 7 or newer, clang++,
  MSVC 2019 and Intel icpx 2021.1.2)

### Sources

The latest version of DUCC can be obtained by cloning the repository via

    git clone https://gitlab.mpcdf.mpg.de/mtr/ducc.git

### Installation

In the following, we assume a Debian-based distribution. For other
distributions, the "apt" lines will need slight changes.

DUCC and its mandatory dependencies can be installed via:

    sudo apt-get install git python3 python3-pip python3-dev python3-pybind11 pybind11-dev
    pip3 install --user git+https://gitlab.mpcdf.mpg.de/mtr/ducc.git

NOTE: compilation of the code will take a significant amount of time
(several minutes). Binary packages are deliberately not made available, since
much better performance can be achieved by compiling the code specifically for
the detected target CPU.


Installing multiple versions simultaneously
-------------------------------------------

The interfaces of the DUCC components are expected to evolve over time; whenever
an interface changes in a manner that is not backwards compatible, the DUCC
version number will increase. As a consequence it might happen that one part of
a Python code may use an older version of DUCC while at the same time another
part requires a newer version. Since DUCC's version number is included in the
module name itself (the module is not called `ducc`, but rather `ducc<X>`),
this is not a problem, as multiple DUCC versions can be installed
simultaneously.
The latest patch levels of a given DUCC version will always be available at the
HEAD of the git branch with the respective name. In other words, if you need
the latest incarnation of DUCC 0, this will be on branch "ducc0" of the
git repository, and it will be installed as the package "ducc0".
Later versions will be maintained on new branches and will be installed as
"ducc1" and "ducc2", so that there will be no conflict with potentially
installed older versions.


DUCC components
===============

ducc.fft
--------

This package provides Fast Fourier, trigonometric and Hartley transforms with a
simple Python interface. It is an evolution of `pocketfft` and `pypocketfft`
which are currently used by `numpy` and `scipy`.

The central algorithms are derived from Paul Swarztrauber's
[FFTPACK](http://www.netlib.org/fftpack) code.

### Features
- supports fully complex and half-complex (i.e. complex-to-real and
  real-to-complex) FFTs, discrete sine/cosine transforms and Hartley transforms
- achieves very high accuracy for all transforms
- supports multidimensional arrays and selection of the axes to be transformed
- supports single, double, and long double precision
- makes use of CPU vector instructions when performing 2D and higher-dimensional
  transforms
- supports prime-length transforms without degrading to O(N**2) performance
- has optional multi-threading support for multidimensional transforms

### Design decisions and performance characteristics
- there is no internal caching of plans and twiddle factors, making the
  interface as simple as possible
- 1D transforms are significantly slower than those provided by FFTW (if FFTW's
  plan generation overhead is ignored)
- multi-D transforms in double precision perform fairly similar to FFTW with
  FFTW_MEASURE; in single precision `ducc.fft` can be significantly faster.

ducc.sht
--------

This package provides efficient spherical harmonic trasforms (SHTs). Its code
is derived from [libsharp](https://arxiv.org/abs/1303.4945), with accelerated
recurrence algorithms presented in
<https://www.jstage.jst.go.jp/article/jmsj/96/2/96_2018-019/_pdf>.


ducc.healpix
------------

This library provides Python bindings for the most important functionality
related to the [HEALPix](https://arxiv.org/abs/astro-ph/0409513) tesselation,
except for spherical harmonic transforms, which are covered by `ducc.sht`.

The design goals are
- similarity to the interface of the HEALPix C++ library
  (while respecting some Python peculiarities)
- simplicity (no optional function parameters)
- low function calling overhead


ducc.totalconvolve
------------------

Library for high-accuracy 4pi convolution on the sphere, which generates a
total convolution data cube from a set of sky and beam `a_lm` and computes
interpolated values for a given list of detector pointings.
This code has evolved from the original
[totalconvolver](https://arxiv.org/abs/astro-ph/0008227) algorithm described
via the [conviqt](https://arxiv.org/abs/1002.1050) code.


### Algorithmic details:
- the code uses `ducc.sht` SHTs and `ducc.fft` FFTs to compute the data cube
- shared-memory parallelization is provided via standard C++ threads.
- for interpolation, the algorithm and kernel described in
  <https://arxiv.org/abs/1808.06736> are used. This allows very efficient
  interpolation with user-adjustable accuracy.


ducc.wgridder
-------------

Library for high-accuracy gridding/degridding of radio interferometry datasets.
An earlier version of this code has been integrated into
[wsclean](https://sourceforge.net/projects/wsclean/)
(<https://arxiv.org/abs/1407.1943>)
as the `wgridder` component.

### Programming aspects
- shared-memory parallelization via standard C++ threads.
- kernel computation is performed on the fly, avoiding inaccuracies
  due to table lookup and reducing overall memory bandwidth

### Numerical aspects
- uses the analytical gridding kernel presented in
  <https://arxiv.org/abs/1808.06736>
- uses the "improved W-stacking method" described in
  <https://arxiv.org/abs/2101.11172>
- in combination these two aspects allow extremely accurate gridding/degridding
  operations (L2 error compared to explicit DFTs can go below 1e-12) with
  reasonable resource consumption


ducc.misc
---------

Various unsorted functionality which will hopefully be categorized in the
future.

