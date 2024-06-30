# MIT License
#
# Copyright (c) 2019 Sze M. Tan, Haoyang Ye, Stephen F Gull, Bojan Nikolic
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import matplotlib.pyplot as plt
import ducc0
from time import time

def calc_map_error(krn, corr, nu, x, W):
    M = len(nu) // W
    nu = nu[:M]
    nx = len(x)
    indx0 = np.arange(M) - (M*(-W+2))
    one_app=np.zeros((nx, M), dtype=np.complex128)
    fact0 = np.exp(2j * np.pi * (- (W / 2) + 1 -nu).reshape((1,-1)) * x.reshape((-1,1)))
    fact1 = np.exp(2j * np.pi * x.reshape((-1,1)))
    for r in range(0, W):
        ell = r - (W / 2) + 1
        indx = indx0 - 2*r*M
        # Use symmetry to deal with negative indices
        indx[indx < 0] = -indx[indx < 0] - 1
        Cr = krn[indx]
        one_app += fact0*Cr
        fact0 *= fact1

    one_app *= corr.reshape((-1,1))
    one_app = (one_app.real-1)**2 + one_app.imag**2
    return np.sum(one_app,  axis=1)/M

def getmaxerr(coeff, W, M, N, x0, D, mach_eps, ofactor):
    nu=(np.arange(W*M)+0.5)/(2*M)
    maxidx = int(2*x0*N+0.9999)+1
    x=np.arange(maxidx)/(2*N)
    krn = ducc0.misc.get_kernel(coeff[0], coeff[1], W, len(nu))
    corr = ducc0.misc.get_correction(coeff[0], coeff[1], W, len(x), 1./(2*N))

    err = calc_map_error(krn, corr, nu, x, W)
    err = np.sqrt(np.max(np.abs(err)))*D
    corr = np.max(corr)/np.min(corr)
    err += mach_eps*corr**D
    return err

def scan_esk(rbeta, re0, x, W, M, N, x0, nsamp, D, mach_eps, ofactor):
    curmin=1e30
    for e0 in np.linspace(re0[0], re0[1], nsamp):
        for beta in np.linspace(rbeta[0], rbeta[1], nsamp):
            test = getmaxerr([beta,e0], W, M, N, x0, D, mach_eps, ofactor)
            if test<curmin:
                curmin, coeffmin = test, [beta,e0]
    return coeffmin, test

def get_best_kernel(D, mach_eps, W, ofactor):
    tol=1e-5
    M=128
    N=512
    x=np.arange(N+1)/(2*N)

    x0 = 0.5/ofactor
    nu=(np.arange(W*M)+0.5)/(2*M)
    rbeta=[1.3, 2.4]
    re0=[0.45, 0.6]
    dbeta = rbeta[1]-rbeta[0]
    de0 = re0[1]-re0[0]
    res = [(re0[0]+re0[1])*0.5, (rbeta[0]+rbeta[1])*0.5]
    err = 1e30
    while dbeta>tol*res[0] or de0>tol*res[1]:
        res_tmp, err_tmp = scan_esk(rbeta, re0, x, W, M, N, x0, 10, D, mach_eps, ofactor)
        if err_tmp < err:
            err = err_tmp
            res = res_tmp
        if dbeta>tol*res[0]:
            dbeta*=0.5
        if de0>tol*res[1]:
            de0*=0.5
        rbeta = [res[0]-0.5*dbeta, res[0]+0.5*dbeta]
        re0 = [res[1]-0.5*de0, res[1]+0.5*de0]

    return res, err

ofactors = np.linspace(1.20,2.50,27)
Ws = np.arange(4,17)
mach_eps = 2.2e-16#1.19e-07
D = 1
results = []
for ofactor in ofactors:
    for W in Ws:
        from time import time
        t0 = time()
        res, err = get_best_kernel(D, mach_eps, W, ofactor)
        print("time:",time()-t0)
        print("{{{0:2d}, {1:4.2f}, {2:13.8g}, {3:19.17f}, {4:20.18f}}},".format(W, ofactor, err, res[0], res[1]))
