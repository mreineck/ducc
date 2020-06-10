#include <functional>
#include "ducc0/math/space_filling.h"
#include "ducc0/infra/error_handling.h"

using namespace std;
using namespace mr;

namespace {

int64_t t00()
  {
  int64_t cnt=0;
  for (uint32_t x=0; x<0x400; ++x)
    for (uint32_t y=0; y<0x400; ++y)
      for (uint32_t z=0; z<0x400; ++z)
        {
        ++cnt;
        auto res = morton2coord3D_32(coord2morton3D_32({x,y,z}));
        MR_assert(res[0]==x && res[1]==y && res[2]==z, "bug");
        }
  return cnt;
  }
int64_t t01()
  {
  int64_t cnt=0;
  for (uint32_t x=0; x<0x400; ++x)
    for (uint32_t y=0; y<0x400; ++y)
      for (uint32_t z=0; z<0x400; ++z)
        {
        ++cnt;
        auto res = block2coord3D_32(coord2block3D_32({x,y,z}));
        MR_assert(res[0]==x && res[1]==y && res[2]==z, "bug");
        }
  return cnt;
  }
int64_t t02()
  {
  int64_t cnt=0;
  for (uint32_t v=0; v<0x40000000; v+=15,++cnt)
    MR_assert (v==peano2morton3D_32(morton2peano3D_32(v,10),10), "bug");
  return cnt;
  }
int64_t t03()
  {
  for (uint32_t v=0; v<0x40000000; ++v)
    MR_assert (v==block2morton3D_32(morton2block3D_32(v)), "bug");
  return 0x40000000;
  }
int64_t t04()
  {
  int64_t cnt=0;
  for (uint32_t v=0; v<0xffffffe0; v+=15,++cnt)
    MR_assert (v==peano2morton2D_32(morton2peano2D_32(v,16),16), "bug");
  return cnt;
  }
int64_t t10()
  {
  int64_t cnt=0;
  for (uint32_t x=0; x<0x10000; ++x)
    for (uint32_t y=0; y<0x10000; ++y)
      {
      ++cnt;
      auto res = morton2coord2D_32(coord2morton2D_32({x,y}));
      MR_assert(res[0]==x && res[1]==y, "bug");
      }
  return cnt;
  }
int64_t t11()
  {
  int64_t cnt=0;
  for (uint32_t x=0; x<0x10000; ++x)
    for (uint32_t y=0; y<0x10000; ++y)
      {
      ++cnt;
      auto res = block2coord2D_32(coord2block2D_32({x,y}));
      MR_assert(res[0]==x && res[1]==y, "bug");
      }
  return cnt;
  }
int64_t t12()
  {
  uint32_t v=0;
  do
    MR_assert (v==block2morton2D_32(morton2block2D_32(v)), "bug");
  while (++v!=0);
  return 0x100000000;
  }
int64_t t20()
  {
  int64_t cnt=0;
  for (uint64_t x=0; x<0xffff0000; x+=0xfff5)
    for (uint64_t y=0; y<0xffff0000; y+=0xfff7)
      {
      ++cnt;
      auto res=morton2coord2D_64(coord2morton2D_64({x,y}));
      MR_assert(res[0]==x && res[1]==y,"bug");
      }
  return cnt;
  }
int64_t t21()
  {
  int64_t cnt=0;
  for (uint64_t x=0; x<0xffff0000; x+=0xfff5)
    for (uint64_t y=0; y<0xffff0000; y+=0xfff7)
      {
      ++cnt;
      auto res = block2coord2D_64(coord2block2D_64({x,y}));
      MR_assert(res[0]==x && res[1]==y,"bug");
      }
  return cnt;
  }
int64_t t22()
  {
  int64_t cnt=0;
  for (uint64_t v=0; v<0xffffffff00000000; v+=0xfffff563, ++cnt)
    MR_assert (v==block2morton2D_64(morton2block2D_64(v)), "bug");
  return cnt;
  }
int64_t t30()
  {
  int64_t cnt=0;
  for (uint64_t x=0; x<0x200000; x+=0xff34)
    for (uint64_t y=0; y<0x200000; y+=0xff84)
      for (uint64_t z=0; z<0x200000; z+=0xff96)
        {
        ++cnt;
        auto res = morton2coord3D_64(coord2morton3D_64({x,y,z}));
        MR_assert(res[0]==x && res[1]==y && res[2]==z, "bug");
        }
  return cnt;
  }
int64_t t31()
  {
  int64_t cnt=0;
  for (uint64_t x=0; x<0x200000; x+=0xff34)
    for (uint64_t y=0; y<0x200000; y+=0xff84)
      for (uint64_t z=0; z<0x200000; z+=0xff96)
        {
        ++cnt;
        auto res = block2coord3D_64(coord2block3D_64({x,y,z}));
        MR_assert(res[0]==x && res[1]==y && res[2]==z, "bug");
        }
  return cnt;
  }
int64_t t32()
  {
  int64_t cnt=0;
  for (uint64_t v=0; v<0x7fffffff00000000; v+=0xfffffff78, ++cnt)
    MR_assert (v==peano2morton3D_64(morton2peano3D_64(v,21),21),"bug");
  return cnt;
  }
int64_t t33()
  {
  int64_t cnt=0;
  for (uint64_t v=0; v<0x7fffffff00000000; v+=0xffffff78, ++cnt)
    MR_assert (v==block2morton3D_64(morton2block3D_64(v)),"bug");
  return cnt;
  }
int64_t t34()
  {
  int64_t cnt=0;
  for (uint64_t v=0; v<0xffffffff00000000; v+=0xfffffff78, ++cnt)
    MR_assert (v==peano2morton2D_64(morton2peano2D_64(v,32),32),"bug");
  return cnt;
  }

} // unnamed namespace

#include <cstdio>
#include "ducc0/infra/timers.h"

using namespace std;

void runtest(function<int64_t()> tf, const char *tn)
  {
  SimpleTimer t;
  auto res = tf();
  printf("%s OK. MOps/s: %7.2f\n",tn,2e-6*res/t());
  }

int main(int argc, const char **argv)
  {
  MR_assert((argc==1)||(argv[0]==nullptr),"problem with args");
  runtest(t10,"coord  <-> Morton 2D 32bit");
  runtest(t11,"coord  <-> Block  2D 32bit");
  runtest(t12,"Morton <-> Block  2D 32bit");
  runtest(t04,"Morton <-> Peano  2D 32bit");
  runtest(t20,"coord  <-> Morton 2D 64bit");
  runtest(t21,"coord  <-> Block  2D 64bit");
  runtest(t22,"Morton <-> Block  2D 64bit");
  runtest(t34,"Morton <-> Peano  2D 64bit");
  runtest(t00,"coord  <-> Morton 3D 32bit");
  runtest(t01,"coord  <-> Block  3D 32bit");
  runtest(t02,"Morton <-> Peano  3D 32bit");
  runtest(t03,"Morton <-> Block  3D 32bit");
  runtest(t30,"coord  <-> Morton 3D 64bit");
  runtest(t31,"coord  <-> Block  3D 64bit");
  runtest(t32,"Morton <-> Peano  3D 64bit");
  runtest(t33,"Morton <-> Block  3D 64bit");
  }
