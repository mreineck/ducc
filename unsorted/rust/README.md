- This provides an initial Rust interface to the C++ ducc code
- The detailed README on ducc can be found [here](https://gitlab.mpcdf.mpg.de/mtr/ducc)
- The Rust wrapper currently supports
  - FFT: c2c
- For the C++ functions that support inplace operations (e.g., the `c2c` FFT), two rust functions are exposed: `ducc0::fft_c2c` and `ducc0::fft_c2c_inplace`.

- Rust wrapper currently *does not* support
  - FFT: c2r, r2c, hartley transforms, ...
  - NuFFT
  - Healpix
  - SHTs
  - Radio response
  - etc.

This wrapper is currently highly experimental. If you encounter problems or need
not yet supported components, please reach out to `c@philipp-arras.de`.

This is my first C++ wrapper for Rust. Feel free to give advice and feedback.
