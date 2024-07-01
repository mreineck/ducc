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

def scan_esk(par_min, par_max, x, W, M, N, x0, nsamp, D, mach_eps):
    bla=ducc0.misc.scan_kernel("esk", par_min, par_max, W, M, N, x0, nsamp, D, mach_eps, nthreads=8)
    return [bla[1], bla[0]]

def get_best_kernel(D, mach_eps, W, ofactor):
    tol=1e-5
    M=128
    N=512
    x=np.arange(N+1)/(2*N)

    x0 = 0.5/ofactor
    nu=(np.arange(W*M)+0.5)/(2*M)
    par_min=[1.3, 0.45]
    par_max=[2.4, 0.6]
#    par_min=[1.3, 0.5]
#    par_max=[2.4, 0.5]
#    par_min=[0.1]
#    par_max=[100]
    dpar = [pmax-pmin for pmin, pmax in zip(par_min, par_max)]
    res = [0.5*(pmax+pmin) for pmin, pmax in zip(par_min, par_max)]
    err = 1e30
    while any([d>tol for d in dpar]):
        res_tmp, err_tmp = scan_esk(par_min, par_max, x, W, M, N, x0, 10, D, mach_eps)
        if err_tmp < err:
            err = err_tmp
            res = res_tmp
        dpar = [0.5*d for d in dpar]
        par_min = [r-0.5*d for r, d in zip(res, dpar)]
        par_max = [r+0.5*d for r, d in zip(res, dpar)]
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
#        print("time:",time()-t0)
#        print(W, ofactor, err, res)
        print("{{{0:2d}, {1:4.2f}, {2:13.8g}, {3:19.17f}, {4:20.18f}}},".format(W, ofactor, err, res[0], res[1]))
