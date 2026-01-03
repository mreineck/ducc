SHT
===

The ``ducc0.sht`` module provides a high-performance interface for transforming data between harmonic space (spherical harmonic coefficients, :math:`a_{\ell m}`) and real space (pixelized maps).

Terminology: Synthesis versus Analysis
--------------------------------------

Users transitioning from other libraries like Healpy often look for ``alm2map`` and ``map2alm`` functions. ``Ducc0`` avoids this naming scheme, because the meaning of ``map2alm`` is not well defined, and because the typical semantics of ``map2alm`` expected by most users are actually ill-defined in most circumstances.
Instead, ``ducc0`` defines a ``synthesis`` operator (corresponding to ``alm2map``) with clearly defined semantics, from which other operators are derived.

1.  **Synthesis**  (:math:`a_{\ell m} \rightarrow` map): This is effectively a **pixelization** process. It takes a continuous signal defined by spherical harmonic coefficients and samples it onto a discrete grid. This operation is a direct linear projection and well defined for any band limit and distribution of pixels over the sphere. Let's denote this as :math:`\mathbf{S}`.
2.  **Adjoint synthesis** (map :math:`\rightarrow a_{\ell m}` ): The transpose operator :math:`\mathbf{S}^\dagger` that maps pixels in real space to spherical harmonics. This operator is **not** the inverse :math:`\mathbf{S}^{-1}`, and in contrast to the inverse has the essential advantage of always being well defined (see below). While not immediately helpful to most end users, this operator is an essential building block for many higher-level (often iterative) algorithms, including approximate map analysis.
3.  **(Pseudo-)Analysis** (map :math:`\rightarrow a_{\ell m}` ): These functions are an **attempt** to explain map values on a given pixelization scheme by a set of :math:`a_{\ell m}` coefficients as closely as possible.
In strong contrast to the synthesis operation, analysis will not be exact in almost all situations, for a series of very different reasons:

- In most real-world scenarios the map contains more pixels than there are degrees of freedom in the corresponding :math:`a_{\ell m}` set.
  Consequently :math:`\mathbf{S}` is not square, making inversion impossible by definition.
- However, for a number of pixelization schemes there exist quadrature rules which in this case (more pixels than harmonic degrees of freedom) at least allow exact recovery of harmonic coefficients from a map that was created by a preceding synthesis operation.

  In other words, ``analysis(synthesis(a_lm)) == a_lm`` will hold in all cases, while the opposite direction ``synthesis(analysis(map))`` will generally **not** be the same as ``map``.

  In yet other words, for some pixelizations, ``analysis`` can be made to work as the "left-inverse" of ``synthesis``, but definitely not as the general inverse.

  ``Ducc`` functions carrying out this kind of operation will contain ``analysis`` in their name, without the ``pseudo``.
- For other pixelizations (HEALPix is a prominent example), not even the left-inverse property can be guaranteed, and the analysis process will be performed by an iterative solver which aims to find a set of ``a_lm`` whose synthesis is as close as possible to the given map in a least-squares sense. Functions performing this task will contain ``pseudo_analysis`` in their name.


Typical workflows
-----------------

Healpix Map Synthesis
~~~~~~~~~~~~~~~~~~~~~

This workflow demonstrates how to generate a pixelized map from a set of :math:`a_{\ell m}` coefficients using :func:`ducc0.sht.synthesis`. This is the standard use case for generating input sky components (CMB, foregrounds…) on a pixel grid::

  import ducc0
  import numpy as np

  lmax = 128
  mmax = lmax

  # We generate a random set of a_ℓm values
  nalm = ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)
  rng = np.random.default_rng(42)
  alm = rng.uniform(-1., 1., nalm) + 1j*rng.uniform(-1., 1., nalm)

  # make a_lm with m==0 real-valued
  alm[0:lmax+1].imag = 0.

  # add an extra leading dimension to the a_ℓm. This is necessary
  # since for transforms with spin≠0, two a_ℓm sets are required
  # instead of one.
  alm = alm.reshape((1,-1))

  hb = ducc0.healpix.Healpix_Base(64, "RING")
  i_map = np.empty((1, hb.npix()))

  # Perform a scalar transformation (spin=0)
  ducc0.sht.synthesis(
      alm=alm,
      map=i_map,
      spin=0,
      lmax=lmax,
      mmax=mmax,
      **hb.sht_info(),
  )

  # Done, the intensity map is i_map[0]


General Map Synthesis
~~~~~~~~~~~~~~~~~~~~~

Instead of pre-computing a pixelated map and interpolating, we can calculate the signal exactly at a given set of positions via :func:`ducc0.sht.synthesis_general`::

  import ducc0
  import numpy as np

  lmax = 128
  mmax = lmax

  # We generate a random set of a_ℓm values
  nalm = ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)
  rng = np.random.default_rng(42)
  alm = rng.uniform(-1., 1., nalm) + 1j*rng.uniform(-1., 1., nalm)

  # make a_lm with m==0 real-valued
  alm[0:lmax+1].imag = 0.

  # add an extra leading dimension to the a_ℓm. This is necessary
  # since for transforms with spin≠0, two a_ℓm sets are required
  # instead of one.
  alm = alm.reshape((1,-1))

  # We want to measure the value of the map along `npos`
  # directions on the 4π sphere
  nloc = 1000
  loc = rng.random((nloc, 2))
  loc[:, 0] *= np.pi           # ϑ (colatitude)
  loc[:, 1] *= 2 * np.pi       # φ (longitude)
  i_values = np.empty((1, nloc))
  values = ducc0.sht.synthesis_general(
      alm=alm,
      spin=0,
      lmax=lmax,
      loc=loc,
      epsilon=1e-8,
  )

  print(values.shape)
  # Prints (1, 1000)


ducc0.sht
---------

.. automodule:: ducc0.sht
    :members:

ducc0.sht.experimental
----------------------

.. automodule:: ducc0.sht.experimental
    :members:
