Unsorted ramblings about lessons learned etc.
=============================================


Avoiding critical strides in multidimensional arrays
----------------------------------------------------

The preferred way to iterate over arrays is in memory order, i.e.
traversing main memory with the smallest possible strides.
However there are situations where array elements need to be accessed
in different ways, e.g. when performing an FFT over noncontiguous
axes of a multi-D array.

This access pattern is inherently slower than access in memory order,
but under specific circumstances it can become especially bad; this
happens whenever the stride of the accesses is a multiple of the
critical stride of the machine (for all practical purposes, this can
be assumed to be 4096 bytes).

``Ducc`` contains helper functions that will avoid critical strides
when allocating multi-D arrays. It achieves this by slightly enlarging
the problematic axes and then returning a sub-view of the allocated
array which has the desired shape.
See the functions ``ducc0.misc.empty_noncritical`` and
``ducc0.misc.make_noncritical``.

More details on critical strides can be found in section 9.10 of
`Agner Fog's optimization manual
<https://agner.org/optimize/optimizing_cpp.pdf>`_.


Initializing newly allocated memory
-----------------------------------

- memory obtained from the system is not initially reserved/allocated,
  just the address space is.

  Case in point: malloc()ing a TB of RAM
  does not crash the system and is actually very fast!

- whenever a memory page in that address space is first "touched"
  (hopefully by a write instruction!), a page fault is generated
  and control is given to the OS, which prepares a physical page
  (including filling it with zeros) to back that part of address space.

  Subsequent accesses to this page take place without any interrupts,
  just as one would expect.

  Page sizes are often 4KB, but can be multi-MB, depending on the
  OS.

- If multiple threads request access to a "new" memory page at the
  same time (e.g. by writing to nearby adresses), this will cause
  lock contention, and the page fault is *much slower* than in the
  single-thread case. This can happen in the inner loops of
  the Legendre transforms.

  This should be avoided at all costs.

  Solution: PAGE_IN tag class for mavs


Prefetching memory
------------------

When accessing large arrays in random order (typically using a
permutation index computed in advance), prefetching may be useful.

The amount of lookahead depends on the application

Solution: mav methods "prefetch_r" and "prefetch_w"


Increasing maximum l multipole for spherical harmonic analysis
--------------------------------------------------------------

See Note.
