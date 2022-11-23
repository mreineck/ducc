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

def gridder_to_C(gridder, W):
    M = len(gridder) // W
    C = np.zeros((W, M), dtype=float)
    for r in range(0, W):
        ell = r - (W / 2) + 1
        indx = (np.arange(M) - 2 * M * ell).astype(int)
        # Use symmetry to deal with negative indices
        indx[indx < 0] = -indx[indx < 0] - 1
        C[r, :] = gridder[indx]
    return C

def calc_map_error_from_C(C, grid_correction, nu_C, x, W):
    M = len(nu_C)
    nx = len(x)
    one_app=np.zeros((nx, M), dtype=np.complex128)
    for r in range(0, W):
        ell = r - (W / 2) + 1
        one_app += grid_correction.reshape((-1,1)) * C[r, :] \
            * np.expm1(2j * np.pi * (ell - nu_C).reshape((1,-1)) * x.reshape((-1,1)))
    one_app +=  grid_correction.reshape((-1,1))  * np.sum(C,axis=0) -1
    one_app = (one_app.real)**2 + one_app.imag**2
    map_error = np.sum(one_app,  axis=1)/M
    return map_error

def calc_map_error(krn, corr, nu, x, W):
    M = len(nu) // W
    C = gridder_to_C(krn, W)
    return calc_map_error_from_C(C, corr, nu[:M], x, W)

def getmaxerr(coeff, nu, W, M, N, x0, D, eps, ofactor):
    nu=(np.arange(W*M)+0.5)/(2*M)
    x=np.arange(N+1)/(2*N)
    krn = ducc0.misc.get_kernel(coeff[0], coeff[1], W, len(nu))
    corr = ducc0.misc.get_correction(coeff[0], coeff[1], W, len(x), 1./(2*N))

    err = calc_map_error(krn, corr, nu, x, W)
    err = np.sqrt(np.max(np.abs(err[0:int(2*x0*N+0.9999)+1])))*D
    corr = np.max(corr[:int(len(corr)/ofactor)+1])/np.min(corr[:int(len(corr)/ofactor)+1])
    err += eps*corr**D
    return err

def scan_esk(rbeta, re0, nu, x, W, M, N, x0, nsamp, D, eps, ofactor):
    curmin=1e30
    for e0 in np.linspace(re0[0], re0[1], nsamp):
        for beta in np.linspace(rbeta[0], rbeta[1], nsamp):
            test = getmaxerr([beta,e0], nu, W, M, N, x0, D, eps, ofactor)
            if test<curmin:
                curmin, coeffmin = test, [beta,e0]
#    print(curmin, coeffmin)
    return coeffmin, test

def scan2(rbeta, re0, nu, x, W, M, N, x0, nsamp, D, eps, ofactor):
    rbeta=[1.9,2.5]
    re0=[0.48, 0.56]
    errors, xbeta, xe0 =[],[],[]
    print(np.average(rbeta), np.std(rbeta))
    print(np.average(re0), np.std(re0))
 #   exit()
    # warm up
    for i in range(1000):
        p1 = np.random.normal(np.average(rbeta), np.std(rbeta))
        p2 = np.random.normal(np.average(re0), np.std(re0))
        test = getmaxerr([p1,p2], nu, W, M, N, x0, D, eps, ofactor)
    #    print(test,p1,p2)
        errors.append(test)
        xbeta.append(p1)
        xe0.append(p2)
    errors=np.array(errors)
    xbeta=np.array(xbeta)
    xe0=np.array(xe0)
    for j in range(1000):
        idx = np.argsort(errors)[:100]
        errors=errors[idx]
        xbeta=xbeta[idx]
        xe0=xe0[idx]
#        print(np.average(xbeta), 3*np.std(xbeta))
#        print(np.average(xe0), 3*np.std(xe0))
        abeta,sbeta=np.average(xbeta),3*np.std(xbeta)
        ae0,se0=np.average(xe0),3*np.std(xe0)
        print(errors[0], xbeta[0], xe0[0], abeta, sbeta, ae0,se0)
        for i in range(1000):
            p1 = np.random.normal(abeta, sbeta)
            p2 = np.random.normal(ae0, se0)
            test = getmaxerr([p1,p2], nu, W, M, N, x0, D, eps, ofactor)
            errors=np.append(errors,test)
            xbeta=np.append(xbeta,p1)
            xe0=np.append(xe0,p2)
    idx = np.argsort(errors)[:100]
    errors=errors[idx]
    xbeta=xbeta[idx]
    xe0=xe0[idx]
    return [xbeta[0], xe0[0]], errors[0]

def get_best_kernel(D, eps, W, ofactor):
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
        res_tmp, err_tmp = scan_esk(rbeta, re0, nu, x, W, M, N, x0, 10, D, eps, ofactor)
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
eps = 2.2e-16#1.19e-07
D = 1
results = []
for ofactor in ofactors:
    for W in Ws:
        res, err = get_best_kernel(D, eps, W, ofactor)
        print("{{{0:2d}, {1:4.2f}, {2:13.8g}, {3:19.17f}, {4:20.18f}}},".format(W, ofactor, err, res[0], res[1]))
