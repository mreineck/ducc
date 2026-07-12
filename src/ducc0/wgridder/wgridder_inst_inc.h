template void ms2dirty<Tcalc, Tacc, Tms, Tms2d_in, Timg>(const cmav<double,2> &uvw,
  const cmav<double,1> &freq, const Tms2d_in &ms,
  const cmav<Tms,2> &wgt_, const cmav<uint8_t,2> &mask_, double pixsize_x, double pixsize_y, double epsilon,
  bool do_wgridding, size_t nthreads, const vmav<Timg,2> &dirty, size_t verbosity,
  bool flip_u, bool flip_v, bool flip_w, bool divide_by_n, double sigma_min,
  double sigma_max, double center_x, double center_y, bool allow_nshift);

template void dirty2ms<Tcalc, Tacc, Tms, Timg>(const cmav<double,2> &uvw,
  const cmav<double,1> &freq, const cmav<Timg,2> &dirty,
  const cmav<Tms,2> &wgt_, const cmav<uint8_t,2> &mask_, double pixsize_x, double pixsize_y,
  double epsilon, bool do_wgridding, size_t nthreads, const vmav<complex<Tms>,2> &ms,
  size_t verbosity, bool flip_u, bool flip_v, bool flip_w, bool divide_by_n,
  double sigma_min, double sigma_max, double center_x, double center_y, bool allow_nshift);

template void ms2dirty_bda<Tcalc, Tacc, Tms, Tms_in, Timg>(
    const cmav<double,2> &uvw,                        // (nrows,3)
    const cmav<size_t,1> &freqlist_id,                // (nrows),
    const cmav<size_t,1> &freqlist_nfreqs,            // (max(freqlist_id)+1)
    const cmav<double,1> &freqlist_freqs,             // (sum(freqlist_nfreqs), concatenated frequency lists for all freqlist_ids
    const Tms_in &ms,
  const cmav<Tms,1> &wgt_, const cmav<uint8_t,1> &mask_, double pixsize_x, double pixsize_y, double epsilon,
  bool do_wgridding, size_t nthreads, const vmav<Timg,2> &dirty, size_t verbosity,
  bool flip_u, bool flip_v, bool flip_w, bool divide_by_n, double sigma_min,
  double sigma_max, double center_x, double center_y, bool allow_nshift);

template void dirty2ms_bda<Tcalc, Tacc, Tms, Timg>(
    const cmav<double,2> &uvw,                        // (nrows,3)
    const cmav<size_t,1> &freqlist_id,                // (nrows),
    const cmav<size_t,1> &freqlist_nfreqs,            // (max(freqlist_id)+1)
    const cmav<double,1> &freqlist_freqs,             // (sum(freqlist_nfreqs), concatenated frequency lists for all freqlist_ids
  const cmav<Timg,2> &dirty,
  const cmav<Tms,1> &wgt_, const cmav<uint8_t,1> &mask_, double pixsize_x, double pixsize_y,
  double epsilon, bool do_wgridding, size_t nthreads, const vmav<complex<Tms>,1> &ms,
  size_t verbosity, bool flip_u, bool flip_v, bool flip_w, bool divide_by_n,
  double sigma_min, double sigma_max, double center_x, double center_y, bool allow_nshift);
