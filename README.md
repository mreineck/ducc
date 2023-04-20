Distinctly Useful Code Collection (DUCC)
========================================

This is a collection of basic programming tools for numerical computation,
including Fast Fourier Transforms, Spherical Harmonic Transforms, non-equispaced
Fourier transforms, as well as some concrete applications like 4pi convolution
on the sphere and gridding/degridding of radio interferometry data.

The code is written in C++17, but provides a simple and comprehensive Python
interface.

### Requirements

- [Python >= 3.7](https://www.python.org/)
- only when compiling from source: [pybind11](https://github.com/pybind/pybind11)
- only when compiling from source: a C++17-capable compiler, e.g.
  - `g++` 7 or later
  - `clang++`
  - MSVC 2019 or later
  - Intel `icpx` (oneAPI compiler series). (Note that the older `icpc` compilers
    are not supported.)

### Sources

The latest version of DUCC can be obtained by cloning the repository via

    git clone https://gitlab.mpcdf.mpg.de/mtr/ducc.git

### Licensing terms
- All source code in this package is released under the terms of the GNU
  General Public License v2 or later.
- Some files (those constituting the FFT component and its internal
  dependencies) are also licensed under the 3-clause BSD license. These files
  contain two sets of licensing headers; the user is free to choose under which
  of those terms they want to use these sources.

### Documentation

Online documentation of the most recent Python interface is available at
https://mtr.pages.mpcdf.de/ducc.

The C++ interface is documented at https://mtr.pages.mpcdf.de/ducc/cpp.
Please note that this interface is not as well documented as the Python one,
and that it should not be considered stable.

### Installation

For best performance, it is recommended to compile DUCC from source, optimizing
for the specific CPU on the system. This can be done using the command

    pip3 install --no-binary ducc0 --user ducc0

NOTE: compilation requires the appropriate compilers to be installed (see above)
and can take a significant amount of time (several minutes).

Alternatively, a simple

    pip3 install --user ducc0

will install a pre-compiled binary package, which makes the installation process
much quicker and does not require any compilers to be installed on the system.
However, the code will most likely perform significantly worse (by a factor of
two to three for some functions) than a custom built version.

Additionally, pre-compiled binaries are distributed for the following systems:

<a href="https://repology.org/project/python:ducc0/versions">
<img src="https://repology.org/badge/vertical-allrepos/python:ducc0.svg" alt="Packaging status">
</a>


<!---
Installing multiple major versions simultaneously
-------------------------------------------------

The interfaces of the DUCC components are expected to evolve over time; whenever
an interface changes in a manner that is not backwards compatible, the DUCC
major version number will increase. As a consequence it might happen that one
part of a Python code may use an older version of DUCC while at the same time
another part requires a newer version. Since DUCC's major version number is
included in the module name itself (the module is not called `ducc`, but rather
`ducc<X>`), this is not a problem, as multiple DUCC versions can be installed
simultaneously.
The latest patch levels of a given DUCC version will always be available at the
HEAD of the git branch with the respective name. In other words, if you need
the latest incarnation of DUCC 0, this will be on branch "ducc0" of the
git repository, and it will be installed as the package "ducc0".
Later versions will be maintained on new branches and will be installed as
"ducc1" and "ducc2", so that there will be no conflict with potentially
installed older versions.
-->

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
- there is no explicit plan management to be done by the user, making the
  interface as simple as possible.
  A small number of plans is cached internally, which does not consume much
  memory, since the storage requirement for a plan only scales with the square
  root of the FFT length for large lengths.
- 1D transforms are significantly slower than those provided by FFTW (if FFTW's
  plan generation overhead is ignored)
- multi-D transforms in double precision perform fairly similar to FFTW with
  FFTW_MEASURE; in single precision `ducc.fft` can be significantly faster.


ducc.nufft
----------

Library for non-uniform FFTs in 1D/2D/3D
(currently only supports transform types 1 and 2).
The goal is to provide similar or better performance and accuracy than
[FINUFFT](https://github.com/flatironinstitute/finufft), making use of lessons
learned during the implementation of the `wgridder` module (see below).


ducc.sht
--------

This package provides efficient spherical harmonic transforms (SHTs). Its code
is derived from [libsharp](https://arxiv.org/abs/1303.4945), but has been
significantly enhanced.

### Noteworthy features
- very efficient support for spherical harmonic synthesis ("alm2map") operations
  and their adjoint for any grid based on iso-latitude rings with equidistant
  pixels in each of the rings.
- support for the same operations on *entirely arbitrary* spherical grids,
  i.e. without constraints on pixel locations. This is implemented via
  intermediate iso-latitude grids and non-uniform FFTs.
- support for accurate spherical harmonic analyis on certain sub-classes of
  grids (Clenshaw-Curtis, Fejer-1 and McEwen-Wiaux) at band limits beyond those
  for which quadrature weights exist. For details see
  [this note](https://wwwmpa.mpa-garching.mpg.de/~martin/shtnote.pdf).
- iterative approximate spherical harmonic analysis on aritrary grids.
- substantially improved transformation speed (up to a factor of 2) on the
  above mentioned grid geometries for high band limits.
- accelerated recurrences as presented in
  [Ishioka (2018)](https://www.jstage.jst.go.jp/article/jmsj/96/2/96_2018-019/_pdf)
- vector instruction support
- multi-threading support

The code for rotating spherical harmonic coefficients was taken (with some
modifications) from Mikael Slevinsky's
[FastTransforms package](https://github.com/MikaelSlevinsky/FastTransforms).


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
[totalconvolver](https://arxiv.org/abs/astro-ph/0008227) algorithm
via the [conviqt](https://arxiv.org/abs/1002.1050) code.


### Algorithmic details:
- the code uses `ducc.sht` SHTs and `ducc.fft` FFTs to compute the data cube
- shared-memory parallelization is provided via standard C++ threads.
- for interpolation, the algorithm and kernel described in
  <https://arxiv.org/abs/1808.06736> are used. This allows very efficient
  interpolation with user-adjustable accuracy.


ducc.wgridder
-------------

Library for high-accuracy gridding/degridding of radio interferometry datasets
(code paper available at <https://arxiv.org/abs/2010.10122>).
This code has also been integrated into
[wsclean](https://gitlab.com/aroffringa/wsclean)
(<https://arxiv.org/abs/1407.1943>)
as the `wgridder` component.

### Programming aspects
- shared-memory parallelization via standard C++ threads.
- kernel computation is performed on the fly, avoiding inaccuracies
  due to table lookup and reducing overall memory bandwidth

### Numerical aspects
- uses a generalization of the analytical gridding kernel presented in
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

This module contains an efficient algorithm for the computation of abscissas and
weights for Gauss-Legendre quadrature. For degrees up to 100, the solutions are
computed in the standard iterative fashion; for higher degrees Ignace Bogaert's
[FastGL algorithm](https://epubs.siam.org/doi/pdf/10.1137/140954969)
is used.
