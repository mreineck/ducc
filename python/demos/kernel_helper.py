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
import ducc0

def get_best_kernel(kernelfunc, D, mach_eps, W, ofactor, par_min, par_max, nthreads):
    # tuning parameters; can most likely be left as they are
    tol=1e-5
    M=128
    N=512
    nsamp=10

    x=np.arange(N+1)/(2*N)

    x0 = 0.5/ofactor
    dpar = [pmax-pmin for pmin, pmax in zip(par_min, par_max)]
    res = [0.5*(pmax+pmin) for pmin, pmax in zip(par_min, par_max)]
    err = 1e30
    # shring the parameter region of interest successively
    while any([d>tol for d in dpar]):
        err_tmp, res_tmp = ducc0.misc.scan_kernel(
            kernelfunc, par_min, par_max, W, M, N,
            x0, nsamp, D, mach_eps, nthreads)
        if err_tmp < err:
            err = err_tmp
            res = res_tmp
        dpar = [0.5*d for d in dpar]
        par_min = [r-0.5*d for r, d in zip(res, dpar)]
        par_max = [r+0.5*d for r, d in zip(res, dpar)]
    return res, err


ofactors = np.linspace(1.20,2.50,27)  # the oversampling factors to consider
Ws = np.arange(4,17)  # range of kernel supports
mach_eps = 2.2e-16  # for double precision; use 1.19e-07 for single precision
D = 1  # dimensionality
nthreads=8

# Standard ES kernel
def kernel(x, par):
    x=np.array(x)
    beta = par[0]*par[1]
    tmp2 = np.abs(x)<=1
    return tmp2*np.exp(beta*(np.sqrt(1-(tmp2*x)**2)-1))

print("Table for standard ES kernels")
for ofactor in ofactors:
    for W in Ws:
        par_min=[1.3, W]
        par_max=[2.4, W]
        res, err = get_best_kernel(kernel, D, mach_eps, W, ofactor, par_min, par_max, nthreads)
        print(W, ofactor, err, res)

# Gauss kernel
def kernel(x, par):
    x=np.array(x)
    sigma = par[0]*par[1]
    tmp2 = np.abs(x)<=1
    return tmp2*np.exp(-sigma*x**2)

print("Table for truncated Gauss kernels")
for ofactor in ofactors:
    for W in Ws:
        par_min=[0.01, W]
        par_max=[100, W]
        res, err = get_best_kernel(kernel, D, mach_eps, W, ofactor, par_min, par_max, nthreads)
        print(W, ofactor, err, res)

# Generalized ES kernel
def kernel(x, par):
    x=np.array(x)
    beta = par[0] * par[2]
    e0 = par[1]
    tmp = (1-x)*(1+x)
    tmp2 = np.abs(x)<=1
    return tmp2*np.exp(beta*((tmp2*tmp)**e0-1))

print("Table for generalized ES kernels")
for ofactor in ofactors:
    for W in Ws:
        par_min=[1.3, 0.45, W]
        par_max=[2.4, 0.6, W]
        res, err = get_best_kernel(kernel, D, mach_eps, W, ofactor, par_min, par_max, nthreads)
        print(W, ofactor, err, res)

