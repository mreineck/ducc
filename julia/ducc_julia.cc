// file ducc_julia.cc

/*
Compilation: (the -I path must point to the src/ directory in the ducc0 checkout)

g++ -O3 -march=native -ffast-math -I ../src/ ducc_julia.cc -Wfatal-errors -pthread -std=c++17 -fPIC -c

Creating the shared library:

g++ -O3 -march=native -o ducc_julia.so ducc_julia.o -Wfatal-errors -pthread -std=c++17 -shared -fPIC
*/

#include "ducc0/infra/threading.cc"
#include "ducc0/infra/mav.cc"
#include "ducc0/math/gl_integrator.cc"
#include "ducc0/math/gridding_kernel.cc"
#include "ducc0/nufft/nufft.h"

using namespace ducc0;
using namespace std;

// typecode stuff

// bits [0-4] contain the type size in bytes
// bit 5 is 1 iff the type is unsigned
// bit 6 is 1 iff the type is floating point
// bit 7 is 1 iff the type is complex-valued
constexpr size_t sizemask = 0x1f;
constexpr size_t unsigned_bit = 0x20;
constexpr size_t floating_point_bit = 0x40;
constexpr size_t complex_bit = 0x80;

template<typename T> class Typecode
  {
  private:
    static constexpr size_t compute()
      {
      static_assert(!is_same<T,bool>::value, "no bools allowed");
//      static_assert(is_arithmetic<T>::value, "need an arithmetic type");
      static_assert(is_integral<T>::value||is_floating_point<T>::value, "need integral or floating point type");
      static_assert(sizeof(T)<=8, "type size must be at most 8 bytes");
      if constexpr(is_integral<T>::value)
        return sizeof(T) + unsigned_bit*(!is_signed<T>::value);
      if constexpr(is_floating_point<T>::value)
        {
        static_assert(is_signed<T>::value, "no support for unsigned floating point types");
        return sizeof(T)+floating_point_bit;
        }
      }
  public:
    static constexpr size_t value = compute();
  };
template<typename T> class Typecode<complex<T>>
  {
  private:
    static constexpr size_t compute()
      {
      static_assert(is_floating_point<T>::value, "need a floating point type");
      return Typecode<T>::value + sizeof(T) + complex_bit;
      }
  public:
    static constexpr size_t value = compute();
  };

constexpr size_t type_size(size_t tc) { return tc&sizemask; }
constexpr bool type_is_unsigned(size_t tc) { return tc&unsigned_bit; }
constexpr bool type_is_floating_point(size_t tc) { return tc&floating_point_bit; }
constexpr bool type_is_complex(size_t tc) { return tc&complex_bit; }

struct ArrayDescriptor
  {
  static constexpr size_t maxdim=5;

  array<uint64_t, maxdim> shape;
  array<int64_t, maxdim> stride;

  void *data;
  uint8_t ndim;
  uint8_t dtype;
  };
template<typename T, size_t ndim> auto prep1(const ArrayDescriptor &desc)
  {
  static_assert(ndim<=ArrayDescriptor::maxdim, "dimensionality too high");
  MR_assert(ndim==desc.ndim, "dimensionality mismatch");
  MR_assert(Typecode<T>::value==desc.dtype, "data type mismatch");
  typename mav_info<ndim>::shape_t shp;
  typename mav_info<ndim>::stride_t str;
  for (size_t i=0; i<ndim; ++i)
    {
    shp[i] = desc.shape[ndim-1-i];
    str[i] = desc.stride[ndim-1-i];
    }
  return make_tuple(shp, str);
  }
template<typename T, size_t ndim> cmav<T,ndim> to_cmav(const ArrayDescriptor &desc)
  {
  auto [shp, str] = prep1<T, ndim>(desc);
  return cmav<T, ndim>(reinterpret_cast<const T *>(desc.data), shp, str);
  }
template<typename T, size_t ndim> vmav<T,ndim> to_vmav(ArrayDescriptor &desc)
  {
  auto [shp, str] = prep1<T, ndim>(desc);
  return vmav<T, ndim>(reinterpret_cast<T *>(desc.data), shp, str);
  }
template<typename T> auto prep2(const ArrayDescriptor &desc)
  {
  MR_assert(Typecode<T>::value==desc.dtype, "data type mismatch");
  typename fmav_info::shape_t shp(desc.ndim);
  typename fmav_info::stride_t str(desc.ndim);
  for (size_t i=0; i<desc.ndim; ++i)
    {
    shp[i] = desc.shape[desc.ndim-1-i];
    str[i] = desc.stride[desc.ndim-1-i];
    }
  return make_tuple(shp, str);
  }
template<typename T> cfmav<T> to_cfmav(const ArrayDescriptor &desc)
  {
  auto [shp, str] = prep2<T>(desc);
  return cfmav<T>(reinterpret_cast<const T *>(desc.data), shp, str);
  }
template<typename T> vfmav<T> to_vfmav(ArrayDescriptor &desc)
  {
  auto [shp, str] = prep2<T>(desc);
  return vfmav<T>(reinterpret_cast<T *>(desc.data), shp, str);
  }

template<typename T> cmav<T,2> get_coord(const ArrayDescriptor &desc)
  {
  auto res(to_cmav<T,2>(desc));
  // flip coord axis!
  return cmav<T,2>(res.data()+(res.shape(1)-1)*res.stride(1),
    res.shape(), {res.stride(0), -res.stride(1)});
  }

extern "C" {

int nufft_u2nu_julia(ArrayDescriptor grid,
                     ArrayDescriptor coord,
                     int forward,
                     double epsilon,
                     size_t nthreads,
                     ArrayDescriptor out,
                     size_t verbosity,
                     double sigma_min,
                     double sigma_max,
                     double periodicity,
                     int fft_order)
  {
  try
    {
    if (coord.dtype==Typecode<double>::value)
      {
      auto mycoord = get_coord<double>(coord);
      if (grid.dtype==Typecode<complex<double>>::value)
        {
        auto mygrid(to_cfmav<complex<double>>(grid));
        auto myout(to_vmav<complex<double>,1>(out));
        MR_assert(mycoord.shape(0)==myout.shape(0), "npoints mismatch");
        MR_assert(mycoord.shape(1)==mygrid.ndim(), "dimensionality mismatch");
        u2nu<double,double>(mycoord,mygrid,forward,epsilon,nthreads,myout,
          verbosity,sigma_min,sigma_max,periodicity,fft_order);
        }
      else if (grid.dtype==Typecode<complex<float>>::value)
        {
        auto mygrid(to_cfmav<complex<float>>(grid));
        auto myout(to_vmav<complex<float>,1>(out));
        MR_assert(mycoord.shape(0)==myout.shape(0), "npoints mismatch");
        MR_assert(mycoord.shape(1)==mygrid.ndim(), "dimensionality mismatch");
        u2nu<float,float>(mycoord,mygrid,forward,epsilon,nthreads,myout,
          verbosity,sigma_min,sigma_max,periodicity,fft_order);
        }
      else
        MR_fail("bad datatype");
      }
    else if (coord.dtype==Typecode<float>::value)
      {
      auto mycoord = get_coord<float>(coord);
      if (grid.dtype==Typecode<complex<float>>::value)
        {
        auto mygrid(to_cfmav<complex<float>>(grid));
        auto myout(to_vmav<complex<float>,1>(out));
        MR_assert(mycoord.shape(0)==myout.shape(0), "npoints mismatch");
        MR_assert(mycoord.shape(1)==mygrid.ndim(), "dimensionality mismatch");
        u2nu<float,float>(mycoord,mygrid,forward,epsilon,nthreads,myout,
          verbosity,sigma_min,sigma_max,periodicity,fft_order);
        }
      else
        MR_fail("bad datatype");
      }
    }
  catch(const exception &e)
    { cout << e.what() << endl; return 1; }
  return 0;
  }

int nufft_nu2u_julia (ArrayDescriptor points,
                       ArrayDescriptor coord,
                       int forward,
                       double epsilon,
                       size_t nthreads,
                       ArrayDescriptor out,
                       size_t verbosity,
                       double sigma_min,
                       double sigma_max,
                       double periodicity,
                       int fft_order)
  {
  try
    {
    if (coord.dtype==Typecode<double>::value)
      {
      auto mycoord = get_coord<double>(coord);
      if (points.dtype==Typecode<complex<double>>::value)
        {
        auto mypoints(to_cmav<complex<double>,1>(points));
        auto myout(to_vfmav<complex<double>>(out));
        MR_assert(mycoord.shape(0)==mypoints.shape(0), "npoints mismatch");
        MR_assert(mycoord.shape(1)==myout.ndim(), "dimensionality mismatch");
        nu2u<double,double>(mycoord,mypoints,forward,epsilon,nthreads,myout,
          verbosity,sigma_min,sigma_max,periodicity,fft_order);
        }
      else if (points.dtype==Typecode<complex<float>>::value)
        {
        auto mypoints(to_cmav<complex<float>,1>(points));
        auto myout(to_vfmav<complex<float>>(out));
        MR_assert(mycoord.shape(0)==mypoints.shape(0), "npoints mismatch");
        MR_assert(mycoord.shape(1)==myout.ndim(), "dimensionality mismatch");
        nu2u<float,float>(mycoord,mypoints,forward,epsilon,nthreads,myout,
          verbosity,sigma_min,sigma_max,periodicity,fft_order);
        }
      else
        MR_fail("bad datatype");
      }
    else if (coord.dtype==Typecode<float>::value)
      {
      auto mycoord = get_coord<float>(coord);
      if (points.dtype==Typecode<complex<float>>::value)
        {
        auto mypoints(to_cmav<complex<float>,1>(points));
        auto myout(to_vfmav<complex<float>>(out));
        MR_assert(mycoord.shape(0)==mypoints.shape(0), "npoints mismatch");
        MR_assert(mycoord.shape(1)==myout.ndim(), "dimensionality mismatch");
        nu2u<float,float>(mycoord,mypoints,forward,epsilon,nthreads,myout,
          verbosity,sigma_min,sigma_max,periodicity,fft_order);
        }
      else
        MR_fail("bad datatype");
      }
    }
  catch(const exception &e)
    { cout << e.what() << endl; return 1; }
  return 0;
  }

struct Tplan
  {
  size_t npoints;
  vector<size_t> shp;
  size_t coord_type;
  void *plan;
  };

Tplan *make_nufft_plan_julia(int nu2u,
                             ArrayDescriptor shape,
                             ArrayDescriptor coord,
                             double epsilon,
                             size_t nthreads,
                             double sigma_min,
                             double sigma_max,
                             double periodicity,
                             int fft_order)
  {
  try
    {
    auto myshape(to_cmav<uint64_t, 1>(shape));
    auto ndim = myshape.shape(0);
    MR_assert(coord.ndim==2, "bad coordinate dimensionality");
    MR_assert(coord.shape[0]==ndim, "dimensionality mismatch");
    auto res = new Tplan{coord.shape[1],vector<size_t>(ndim),coord.dtype,nullptr};
    for (size_t i=0; i<ndim; ++i)
      res->shp[i] = myshape(ndim-1-i);
    if (coord.dtype==Typecode<double>::value)
      {
      auto mycoord = get_coord<double>(coord);
      if (ndim==1)
        res->plan = new Nufft<double, double, double, 1>(nu2u, mycoord, array<size_t,1>{myshape(0)},
          epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
      else if (ndim==2)
        res->plan = new Nufft<double, double, double, 2>(nu2u, mycoord, array<size_t,2>{myshape(1),myshape(0)},
          epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
      else if (ndim==3)
        res->plan = new Nufft<double, double, double, 3>(nu2u, mycoord, array<size_t,3>{myshape(2),myshape(1),myshape(0)},
          epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
      else
        MR_fail("bad number of dimensions");
      return res;
      }
    else if (coord.dtype==Typecode<float>::value)
      {
      auto mycoord = get_coord<float>(coord);
      if (ndim==1)
        res->plan = new Nufft<float, float, float, 1>(nu2u, mycoord, array<size_t,1>{myshape(0)},
          epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
      else if (ndim==2)
        res->plan = new Nufft<float, float, float, 2>(nu2u, mycoord, array<size_t,2>{myshape(1),myshape(0)},
          epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
      else if (ndim==3)
        res->plan = new Nufft<float, float, float, 3>(nu2u, mycoord, array<size_t,3>{myshape(2),myshape(1),myshape(0)},
          epsilon, nthreads, sigma_min, sigma_max, periodicity, fft_order);
      else
        MR_fail("bad number of dimensions");
      return res;
      }
    MR_fail("bad coordinate data type");
    }
  catch(const exception &e)
    { cout << e.what() << endl; return nullptr; }
  }

int delete_nufft_plan_julia(Tplan *plan)
  {
  try
    {
    if (plan->shp.size()==1)
      (plan->coord_type==Typecode<double>::value) ?
        delete reinterpret_cast<Nufft<double, double, double, 1> *>(plan->plan)
      : delete reinterpret_cast<Nufft<float,float,float, 1> *>(plan->plan);
    else if (plan->shp.size()==2)
      (plan->coord_type==Typecode<double>::value) ?
        delete reinterpret_cast<Nufft<double, double, double, 2> *>(plan->plan)
      : delete reinterpret_cast<Nufft<float,float,float, 2> *>(plan->plan);
    else if (plan->shp.size()==3)
      (plan->coord_type==Typecode<double>::value) ?
        delete reinterpret_cast<Nufft<double, double, double, 2> *>(plan->plan)
      : delete reinterpret_cast<Nufft<float,float,float, 2> *>(plan->plan);
    else
      MR_fail("bad number of dimensions");
    delete plan;
    }
  catch(const exception &e)
    { cout<< e.what() << endl; return 1; }
  return 0;
  }

int planned_nu2u_julia(Tplan *plan, int forward, size_t verbosity,
  ArrayDescriptor points, ArrayDescriptor uniform)
  {
  try
    {
    MR_assert(uniform.ndim==plan->shp.size(), "dimensionality mismatch");
    for (size_t i=0; i<uniform.ndim; ++i)
      MR_assert(uniform.shape[i]==plan->shp[i], "array dimension mismatch");
    if (points.dtype==Typecode<complex<double>>::value)
      {
      MR_assert(plan->coord_type==Typecode<double>::value, "data type mismatch");
      auto mypoints(to_cmav<complex<double>,1>(points));
      if (plan->shp.size()==1)
        {
        auto myout(to_vmav<complex<double>,1>(uniform));
        auto rplan = reinterpret_cast<Nufft<double, double, double, 1> *>(plan->plan);
        rplan->nu2u(forward, verbosity, mypoints, myout);
        }
      else if (plan->shp.size()==2)
        {
        auto myout(to_vmav<complex<double>,2>(uniform));
        auto rplan = reinterpret_cast<Nufft<double, double, double, 2> *>(plan->plan);
        rplan->nu2u(forward, verbosity, mypoints, myout);
        }
      else if (plan->shp.size()==3)
        {
        auto myout(to_vmav<complex<double>,3>(uniform));
        auto rplan = reinterpret_cast<Nufft<double, double, double, 3> *>(plan->plan);
        rplan->nu2u(forward, verbosity, mypoints, myout);
        }
      else
        MR_fail("bad number of dimensions");
      }
    else if (points.dtype==Typecode<complex<float>>::value)
      {
      MR_assert(plan->coord_type==Typecode<float>::value, "data type mismatch");
      auto mypoints(to_cmav<complex<float>,1>(points));
      if (plan->shp.size()==1)
        {
        auto myout(to_vmav<complex<float>,1>(uniform));
        auto rplan = reinterpret_cast<Nufft<float, float, float, 1> *>(plan->plan);
        rplan->nu2u(forward, verbosity, mypoints, myout);
        }
      else if (plan->shp.size()==2)
        {
        auto myout(to_vmav<complex<float>,2>(uniform));
        auto rplan = reinterpret_cast<Nufft<float, float, float, 2> *>(plan->plan);
        rplan->nu2u(forward, verbosity, mypoints, myout);
        }
      else if (plan->shp.size()==3)
        {
        auto myout(to_vmav<complex<float>,3>(uniform));
        auto rplan = reinterpret_cast<Nufft<float, float, float, 3> *>(plan->plan);
        rplan->nu2u(forward, verbosity, mypoints, myout);
        }
      else
        MR_fail("bad number of dimensions");
      }
    else
      MR_fail("unsupported data type");
    }
  catch(const exception &e)
    { cout<< e.what() << endl; return 1; }
  return 0;
  }

int planned_u2nu_julia(Tplan *plan, int forward, size_t verbosity,
  ArrayDescriptor uniform, ArrayDescriptor points)
  {
  try
    {
    MR_assert(uniform.ndim==plan->shp.size(), "dimensionality mismatch");
    for (size_t i=0; i<uniform.ndim; ++i)
      MR_assert(uniform.shape[i]==plan->shp[i], "array dimension mismatch");
    if (points.dtype==Typecode<complex<double>>::value)
      {
      MR_assert(plan->coord_type==Typecode<double>::value, "data type mismatch");
      auto mypoints(to_vmav<complex<double>,1>(points));
      if (plan->shp.size()==1)
        {
        auto myuniform(to_cmav<complex<double>,1>(uniform));
        auto rplan = reinterpret_cast<Nufft<double, double, double, 1> *>(plan->plan);
        rplan->u2nu(forward, verbosity, myuniform, mypoints);
        }
      else if (plan->shp.size()==2)
        {
        auto myuniform(to_cmav<complex<double>,2>(uniform));
        auto rplan = reinterpret_cast<Nufft<double, double, double, 2> *>(plan->plan);
        rplan->u2nu(forward, verbosity, myuniform, mypoints);
        }
      else if (plan->shp.size()==3)
        {
        auto myuniform(to_cmav<complex<double>,3>(uniform));
        auto rplan = reinterpret_cast<Nufft<double, double, double, 3> *>(plan->plan);
        rplan->u2nu(forward, verbosity, myuniform, mypoints);
        }
      else
        MR_fail("bad number of dimensions");
      }
    else if (points.dtype==Typecode<complex<float>>::value)
      {
      MR_assert(plan->coord_type==Typecode<float>::value, "data type mismatch");
      auto mypoints(to_vmav<complex<float>,1>(points));
      if (plan->shp.size()==1)
        {
        auto myuniform(to_cmav<complex<float>,1>(uniform));
        auto rplan = reinterpret_cast<Nufft<float, float, float, 1> *>(plan->plan);
        rplan->u2nu(forward, verbosity, myuniform, mypoints);
        }
      else if (plan->shp.size()==2)
        {
        auto myuniform(to_cmav<complex<float>,2>(uniform));
        auto rplan = reinterpret_cast<Nufft<float, float, float, 2> *>(plan->plan);
        rplan->u2nu(forward, verbosity, myuniform, mypoints);
        }
      else if (plan->shp.size()==3)
        {
        auto myuniform(to_cmav<complex<float>,3>(uniform));
        auto rplan = reinterpret_cast<Nufft<float, float, float, 3> *>(plan->plan);
        rplan->u2nu(forward, verbosity, myuniform, mypoints);
        }
      else
        MR_fail("bad number of dimensions");
      }
    else
      MR_fail("unsupported data type");
    }
  catch(const exception &e)
    { cout<< e.what() << endl; return 1; }
  return 0;
  }

}
