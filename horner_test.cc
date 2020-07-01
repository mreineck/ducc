// compile with "g++ -O3 -ffast-math -std=c++17 -march=native -Isrc horner_test.cc"

#include "ducc0/math/horner_kernel.h"
#include "ducc0/infra/timers.h"
#include <iostream>

using namespace ducc0;
using namespace std;


// you can use any kernel defined on [-1; 1] here
double es_kernel(size_t w, double x)
  {
  auto beta = 2.3*w;
  return exp(beta*(sqrt(1-x*x)-1));
  }


int main()
  {
  constexpr size_t W=12; // kernel support in pixels
  constexpr size_t D=12; // degree of approximating polynomials
  using FLT=double;

  size_t Neval = 1000000000; // we will do a billion function evaluations altogether 
  size_t Ncall = Neval/W;
  FLT delta = 2./W/double(Ncall); // small offsets between individual calls to prevent the compiler from optimizing too much

  HornerKernel<W,D,FLT> hk([](double x){return es_kernel(W,x);});
  double sum=0;
  SimpleTimer timer;
  for (size_t i=0; i<Ncall; ++i)
    {
    FLT p0 = -FLT(1)+i*delta;  // position of first sample
    auto res = hk.eval(p0);

    for (size_t i=0; i<W; ++i)
      sum +=res[i]; // needed to avoid over-optimization
    }
  cout << "HornerKernel: " << Neval/timer() << " function approximations per second" << endl;
  cout << sum << endl;

  HornerKernelFlexible<FLT> hk2(W,D,[](double x){return es_kernel(W,x);});
  sum=0;
  timer.reset();
  for (size_t i=0; i<Ncall; ++i)
    {
    FLT p0 = -FLT(1)+i*delta;  // position of first sample
    auto res = hk2.eval(p0);

    for (size_t i=0; i<W; ++i)
      sum +=res[i]; // needed to avoid over-optimization
    }
  cout << "HornerKernelFlexible: " << Neval/timer() << " function approximations per second" << endl;
  cout << sum << endl;
  }
