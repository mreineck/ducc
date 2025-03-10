template void ms2dirty<Tcalc, Tacc, Tms, Tms_in, Timg>(const cmav<double,2> &uvw,
  const cmav<double,1> &freq, const Tms_in &ms,
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

template void ms2dirty_tuning<Tcalc, Tacc, Tms, Timg, Tms_in>(const cmav<double,2> &uvw,
  const cmav<double,1> &freq, const Tms_in &ms,
  const cmav<Tms,2> &wgt_, const cmav<uint8_t,2> &mask_, double pixsize_x, double pixsize_y, double epsilon,
  bool do_wgridding, size_t nthreads, const vmav<Timg,2> &dirty, size_t verbosity,
  bool flip_u, bool flip_v, bool flip_w, bool divide_by_n, double sigma_min,
  double sigma_max, double center_x, double center_y);

template void dirty2ms_tuning<Tcalc, Tacc, Tms, Timg>(const cmav<double,2> &uvw,
  const cmav<double,1> &freq, const cmav<Timg,2> &dirty,
  const cmav<Tms,2> &wgt_, const cmav<uint8_t,2> &mask_, double pixsize_x, double pixsize_y,
  double epsilon, bool do_wgridding, size_t nthreads, const vmav<complex<Tms>,2> &ms,
  size_t verbosity, bool flip_u, bool flip_v, bool flip_w, bool divide_by_n,
  double sigma_min, double sigma_max, double center_x, double center_y);
