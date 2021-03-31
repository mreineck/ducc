#ifndef MRBASE_VEC_H
#define MRBASE_VEC_H

#include <array>
#include <cmath>
#include <type_traits>
#include <iostream>

/* Very simple templated 1/2/3D vector of arbitrary floating-point type.
   Supports member access via indices.
   Supports vector addition, subtraction, scaling, dot product, cross product
   (for 3D only), length calculation and output to a stream.
   Most incorrect uses are caught at compile time. */
template<typename T, int dim> class vec
  {
  private:
    static_assert((dim>=1) && (dim<=3), "vec dimensionality must be in [1; 3]");
    static_assert(std::is_floating_point<T>::value,
      "vec must be built on a floating point type");

    std::array<T, dim> d;

  public:
    vec() {}
    vec(const T &x_)
      : d {x_} { static_assert(dim==1, "dim must be 1"); }
    template<typename U=T>
      vec(const T &x_, const T &y_)
      : d {x_, y_} { static_assert(dim==2, "dim must be 2"); }
    template<typename U=T>
      vec(const T &x_, const T &y_, const T &z_)
      : d {x_, y_, z_} { static_assert(dim==3, "dim must be 3"); }

    template<typename T2> explicit vec(const vec<T2, dim> &v)
      {for (int i=0; i<dim; ++i) d[i] = v.d[i];}

    const T &operator[](int i) const { return d[i]; }
    T &operator[](int i) { return d[i]; }

    vec operator+(const vec &v) const
      {
      vec res;
      for (int i=0; i<dim; ++i) res[i] = d[i]+v.d[i];
      return res;
      }
    vec &operator+=(const vec &v)
      {
      for (int i=0; i<dim; ++i) d[i]+=v.d[i];
      return *this;
      }
    vec operator-(const vec &v) const
      {
      vec res;
      for (int i=0; i<dim; ++i) res[i] = d[i]-v.d[i];
      return res;
      }
    vec &operator-=(const vec &v)
      {
      for (int i=0; i<dim; ++i) d[i]-=v.d[i];
      return *this;
      }
    vec operator*(const T &fct) const
      {
      vec res;
      for (int i=0; i<dim; ++i) res[i] = d[i]*fct;
      return res;
      }
    vec &operator*=(const T &fct)
      {
      for (int i=0; i<dim; ++i) d[i]*=fct;
      return *this;
      }
    T dot(const vec &v) const
      {
      T res = d[0]*v.d[0];
      for (int i=1; i<dim; ++i) res += d[i]*v.d[i];
      return res;
      }
    T squaredLength() const
      {
      T res = d[0]*d[0];
      for (int i=1; i<dim; ++i) res += d[i]*d[i];
      return res;
      }
    T length() const
      { return std::sqrt(squaredLength()); }
    vec cross(const vec &v)
      {
      static_assert(dim==3, "dim must be 3");
      return vec(d[1]*v.d[2] - d[2]*v.d[1],
                 d[2]*v.d[0] - d[0]*v.d[2],
                 d[0]*v.d[1] - d[1]*v.d[0]);
      }
    };

template<typename T, int dim> inline vec<T, dim> operator*(const T &fct,
  const vec<T, dim> &v)
    { return v*fct; }

template<typename T, int dim> inline std::ostream &operator<<
  (std::ostream &os, const vec<T, dim> &v)
  {
  os << v[0];
  for (int i=1; i<dim; ++i) os << " " << v[i];
  return os;
  }

#endif
