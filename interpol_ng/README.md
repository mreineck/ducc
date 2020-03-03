interpol_ng
===========

Library for high-accuracy 4pi convolution on the sphere, which generates a
total convolution data cube from a set of sky and beam `a_lm` and computes
interpolated values for a given list of detector pointings.

Algorithmic details:
- the code uses `libsharp` SHTs to compute the data cube
- shared-memory parallelization is provided via standard C++ threads.
- for interpolation, the algorithm and kernel described in
  https://arxiv.org/abs/1808.06736 are used. This allows very efficient
  interpolation with user-adjustable accuracy.

Installation and prerequisites:
- execute `python3 setup.py install --user`. This requires the `g++` compiler.
  If you want to compile with `clang`, use
  `CC="clang -fsized-deallocation" python3 setup.py install --user`
- `numpy` and `pybind11` are required for the Python interface.
- For tests and demos, the package `pysharp` (also available in this repository)
  is required.
- For the unit tests, `pytest` is required.
