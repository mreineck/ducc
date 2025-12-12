SHT
===

The ``ducc0.sht`` module provides a high-performance interface for transforming data between harmonic space (spherical harmonic coefficients, :math:`a_{\ell m}`) and real space (pixelized maps).

Terminology: Synthesis versus Analysis
--------------------------------------

Users transitioning from other libraries like Healpy often look for ``alm2map`` and ``map2alm``. ``ducc0`` avoids this naming scheme to highlight a fundamental mathematical asymmetry between the two operations:

1.  **Synthesis**  (:math:`a_{\ell m} \rightarrow` Map): This is effectively a **pixelization** process. It takes a continuous signal defined by spherical harmonic coefficients and samples it onto a discrete grid. This operation is a direct linear projection and is exact, although limited by floating-point precision.

2.  **Analysis** (Map :math:`\rightarrow a_{\ell m}`): This is an **estimation** process that attempts to recover the continuous coefficients from a finite set of discrete pixels. Since the map is a discretized approximation of the signal on the sphere, this is often an ill-posed inverse problem. Recovering the exact input :math:`a_{\ell m}` typically requires iterative fitting or precise quadrature weights, rather than a simple matrix multiplication.

Because of this asymmetry, the two operations are not strictly invertible: ``analysis(synthesis(alm))`` is not guaranteed to return the exact input ``alm`` due to pixelization window functions and sampling limits.

“Adjoint” vs. “Inverse”
~~~~~~~~~~~~~~~~~~~~~~~

In ``ducc0``, you will frequently see functions with the ``adjoint_`` prefix. It is important to distinguish the mathematical *adjoint* (transpose) from the *inverse*.

- **Synthesis**: The forward operator :math:`\mathbf{S}` that maps the harmonic space into the real space of pixels
- **Adjoint synthesis**: The transpose operator :math:`\mathbf{S}^\dagger` that maps pixels in real space to spherical harmonics. This operator is **not** the inverse :math:`\mathbf{S}^{-1}`.

However, the adjoint is the core building block for solving the inverse problem. If you want to estimate the spherical harmonic coefficients :math:`a_{\ell m}` from a map, you generally have two paths:

1.  **Direct analysis** (:func:`ducc0.sht.analysis_2d`). These apply quadrature weights to approximate the integral transform. This is fast, and it can be accurate if you use proper grid schemes.

2.  **Iterative estimation** (:func:`ducc0.sht.pseudo_analysis`, :func:`ducc0.sht.pseudo_analysis_general`, or custom solvers). These use the ``adjoint_synthesis`` algorithm (:func:`ducc0.sht.adjoint_synthesis`, :func:`ducc0.sht.adjoint_synthesis_2d`, :func:`ducc0.sht.adjoint_synthesis_general`) repeatedly to solve for the :math:`a_{\ell m}` that best fit the data in a least-squares sense.

As a rule of thumb, if you want to pixelize a sky model in the form of :math:`a_{\ell m}`, use ``synthesis``. If you want to estimate :math:`a_{\ell m}` from data, look for ``analysis*`` to get an approximation or ``adjoint_synthesis*`` if you are building a solver.


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
