FFT
===

Miscellaneous notes on implementation details
---------------------------------------------

1. Algorithm to determine good FFT sizes

   - Finding a smooth number >= n
   
     For complex transforms, optimal FFT lengths are composites of
     2, 3, 5, 7, 11; for real transforms they should be composites of
     2, 3, 5 only.
   
     First, we need an upper limit for the desired number; we can
     use 2*n here, since between n and 2*n there will always be a power of
     2, which by definition fulfils the smoothness criterion.
   
     To actually find the best candidate, we search for the smallest
     composite >= n below the current limit using a set of nested loops,
     each taking care of one of the allowed prime numbers, in descending
     size from outermost to innermost loops.
   
     The innermost couple of loops (dealing with factors 2 and 3) is fused
     into a single one following a clever algorithm by Peter Bell, to
     minimize the total iteration count.
   
     Whenever a candidate is found, the current limit is set to that
     number, further reducing subsequent iterations.
   
     Should n be found to be a candidate, it is returned immediately.
   
   - Enforcing a specific factor to be present in the target size
   
     Sometimes it is essential that the returned FFT size is even, or
     perhaps a multiple of the system's SIMD vector length. In such
     situations the required factor can be specified, and the code will
     enforce its presence by computing
   
     good_size = required_factor * smooth_size(n+required_factor-1)/required_factor)
   
     (where "/" represents integer division).


ducc0.fft
---------

.. automodule:: ducc0.fft
    :members:
