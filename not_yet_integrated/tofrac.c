#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

/* f : number to convert.
 * num, denom: returned parts of the rational.
 * md: max denominator value.  Note that machine floating point number
 *     has a finite resolution (10e-16 ish for 64 bit double), so specifying
 *     a "best match with minimal error" is often wrong, because one can
 *     always just retrieve the significand and return that divided by
 *     2**52, which is in a sense accurate, but generally not very useful:
 *     1.0/7.0 would be "2573485501354569/18014398509481984", for example.
 */
void rat_approx0(double f, int64_t md, int64_t *num, int64_t *denom)
  {
  /*  a: continued fraction coefficients. */
  int64_t h[3] = { 0, 1, 0 }, k[3] = { 1, 0, 0 };

  if (md <= 1) { *denom = 1; *num = (int64_t) f; return; }

  int neg = 0;
  if (f < 0) { neg = 1; f = -f; }

  int64_t n = 1;
  while (f != floor(f)) { n <<= 1; f *= 2; }
  int64_t d = f;

  /* continued fraction and check denominator each step */
  for (int i = 0; i < 64; i++) {
  	int64_t a = n ? d / n : 0;
  	if (i && !a) break;

  	int64_t x = d; d = n; n = x % n;

  	x = a;
  	if (k[1] * a + k[0] >= md) {
  		x = (md - k[0]) / k[1];
  		if (x * 2 >= a || k[1] >= md)
  			i = 65;
  		else
  			break;
  	}

  	h[2] = x * h[1] + h[0]; h[0] = h[1]; h[1] = h[2];
  	k[2] = x * k[1] + k[0]; k[0] = k[1]; k[1] = k[2];
  }
  *denom = k[1];
  *num = neg ? -h[1] : h[1];
}
void rat_approx(double f, int64_t md, int64_t *num, int64_t *denom)
{
	/*  a: continued fraction coefficients. */
	int64_t a, h[3] = { 0, 1, 0 }, k[3] = { 1, 0, 0 };
	int64_t x, d, n = 1;
	int i, neg = 0;

	if (md <= 1) { *denom = 1; *num = (int64_t) f; return; }

	if (f < 0) { neg = 1; f = -f; }

	while (f != floor(f)) { n <<= 1; f *= 2; }
	d = f;

	/* continued fraction and check denominator each step */
	for (i = 0; i < 64; i++) {
		a = n ? d / n : 0;
		if (i && !a) break;

		x = d; d = n; n = x % n;

		x = a;
		if (k[1] * a + k[0] >= md) {
			x = (md - k[0]) / k[1];
			if (x * 2 >= a || k[1] >= md)
				i = 65;
			else
				break;
		}

		h[2] = x * h[1] + h[0]; h[0] = h[1]; h[1] = h[2];
		k[2] = x * k[1] + k[0]; k[0] = k[1]; k[1] = k[2];
	}
	*denom = k[1];
	*num = neg ? -h[1] : h[1];
}
int tofrac(double x, size_t maxden, size_t *nom, size_t *den)
  {
  double startx = x;

  /* initialize matrix */
  long m[2][2];
  m[0][0] = m[1][1] = 1;
  m[0][1] = m[1][0] = 0;

  /* loop finding terms until denom gets too big */
  long ai;
  while (m[1][0]*(ai=(long)x) + m[1][1] <= maxden)
    {
    long t = m[0][0] * ai + m[0][1];
    m[0][1] = m[0][0];
    m[0][0] = t;
    t = m[1][0] * ai + m[1][1];
    m[1][1] = m[1][0];
    m[1][0] = t;
    if(x==(double)ai) {printf("1\n");break;}     // AF: division by zero
    x = 1/(x - (double) ai);
    if(x>(double)0x7FFFFFFF) {printf("2\n");break;}  // AF: representation failure
    }

  *nom=m[0][0]; *den=m[1][0];
  if (fabs(x-(*nom/ *den))<(1e-15*fabs(x)))
    return 1;
  else
    return 0;
  /* now remaining x is between 0 and 1/ai */
  /* approx as either 0 or 1/m where m is max that will fit in maxden */
  /* first try zero */
  printf("%ld/%ld, error = %e\n", m[0][0], m[1][0],
         startx - ((double) m[0][0] / (double) m[1][0]));
  }

int main(void)
  {
  size_t num, den;
  rat_approx(-27./67564.-1e-14,1<<30,&num, &den);
  printf ("%d %d\n",(int)num,(int)den);
  }
