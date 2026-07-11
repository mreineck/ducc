Miscellaneous
=============

ducc0.misc
----------

Miscellaneous notes
-------------------

1. Gauss-Legendre integration

   For problems with more than 100 abscissas, ducc0 uses Ignace Bogaert's
   "FastGL" approach for determining the Gauss-Legendre abscissas
   and weights (https://epubs.siam.org/doi/pdf/10.1137/140954969).

   Since this algorithm (at least in this particular form) doesn't work
   for smaller problems, ducc0 switches to the traditional iterative
   refinement approach for problems with fewer abscissas. This is still
   very efficient, but ducc0 also caches all of the abscissas and
   weights computed this way, which makes subsequent calls to the
   integrator functions even faster.

2. Prolate spheroidal wave functions

   Ducc0 can compute prolate spheroidal wave functions (PSWF) of order 0,
   which are useful for non-uniform FFTs, radio interferometry gridding,
   and many more contexts.

   The implementation is based on Vladimir Rokhlin's Fortran code,
   available in https://github.com/flatironinstitute/dmk.
   Several performance improvements are added on top:

   - Since the functions in question are even, we can skip all
     polynomial coefficients of odd degree. This allows for
     significantly faster evaluation and reduces storage requirements.
   - The function evaluation routine avoids all floating-point division
     instructions and uses small tables of precomputed values where
     necessary, which results in another considerable speed-up.
   - Further gains are possible by computing several function values
     simultaneously, by using SIMD instructions.

   Overall, setting up a PSWF computation object for a given parameter
   ``c`` requires (roughly) ``c`` microseconds, whereas any subsequent
   function evaluation takes 10-100 nanoseconds.


.. automodule:: ducc0.misc
    :members:

ducc0.misc.experimental
-----------------------

.. automodule:: ducc0.misc.experimental
    :members:
