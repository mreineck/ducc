#ifndef MRUTIL_VECSUPPORT_H
#define MRUTIL_VECSUPPORT_H

#include <cstdlib>
#include <cmath>

namespace mr {

namespace vecsupport {

namespace detail {

using namespace std;

#if (defined(__AVX512F__))
constexpr size_t vbytes = 64;
#elif (defined(__AVX__))
constexpr size_t vbytes = 32;
#elif (defined(__SSE2__))
constexpr size_t vbytes = 16;
#elif (defined(__VSX__))
constexpr size_t vbytes = 16;
#endif

template<typename T, size_t len=vbytes/sizeof(T)> class vtp
  {
  protected:
    using Tv __attribute__ ((vector_size (len*sizeof(T)))) = T;
    static_assert((len>0) && ((len&(len-1))==0), "bad vector length");
    Tv v;

    void from_scalar(T other)
      { for (size_t i=0; i<len; ++i) v[i]=other; }

  public:
    static constexpr size_t vlen=len;
    vtp () {}
    vtp(T other)
      { from_scalar(other); }
    vtp(const Tv &other)
      : v(other) {}
    vtp(const vtp &other) = default;
    vtp &operator=(T other)
      { from_scalar(other); return *this; }
    vtp operator-() const { return vtp(-v); }
    vtp operator+(vtp other) const { return vtp(v+other.v); }
    vtp operator-(vtp other) const { return vtp(v-other.v); }
    vtp operator*(vtp other) const { return vtp(v*other.v); }
    vtp operator/(vtp other) const { return vtp(v/other.v); }
    vtp &operator+=(vtp other) { v+=other.v; return *this; }
    vtp &operator-=(vtp other) { v-=other.v; return *this; }
    vtp &operator*=(vtp other) { v*=other.v; return *this; }
    vtp &operator/=(vtp other) { v/=other.v; return *this; }
    vtp exp() const
      {
      vtp res;
      for (size_t i=0; i<len; ++i) res.v[i] = std::exp(v[i]);
      return res;
      }
    vtp sqrt() const
      {
      vtp res;
      for (size_t i=0; i<len; ++i) res.v[i] = std::sqrt(v[i]);
      return res;
      }

    template<typename I> void Set (I i, T val) { v[i]=val; }
    template<typename I> T operator[](I i) const { return v[i]; }
  };

}

using detail::vtp;

}}

#endif
