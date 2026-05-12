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

.. automodule:: ducc0.misc
    :members:

ducc0.misc.experimental
-----------------------

.. automodule:: ducc0.misc.experimental
    :members:
