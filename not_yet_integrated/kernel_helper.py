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

def C_to_grid_correction(C, nu_C, x, optimal=True):
    W = C.shape[0]
    nx = x.shape[0]
    c = np.zeros(x.shape, dtype=float)
    d = np.zeros(x.shape, dtype=float)
    C = C.reshape((C.shape[0],C.shape[1],1))
    nu_C = nu_C.reshape((-1,1))
    x = x.reshape((1,-1))
    cosarr = np.empty((W,nx))
    for i in range(W):
        cosarr[i,:] = np.cos(2 * np.pi * i * x)
    for rp in range(0, W):
        ellp = rp - (W / 2) + 1
        for r in range(0, W):
            ell = r - (W / 2) + 1
            xmn = np.mean(C[rp, :, :]*C[r, :, :], axis=0)
            tmp = xmn * cosarr[abs(rp-r)]
            d += tmp
        tmp2 = C[rp, :, :] * np.cos(2 * np.pi * (ellp-nu_C) * x)
        c += np.mean(tmp2,axis=0)
    return c / d if optimal else 1 / c


def gridder_to_grid_correction(gridder, nu, x, W, optimal=True):
    M = len(nu) // W
    C = gridder_to_C(gridder, W)
    return C_to_grid_correction(C, nu[:M], x, optimal)


def calc_map_error_from_C(C, grid_correction, nu_C, x, W):
    M = len(nu_C)
    nx = len(x)
    one_app=np.zeros((nx, M), dtype=np.complex128)
    for r in range(0, W):
        ell = r - (W / 2) + 1
        one_app += grid_correction.reshape((-1,1)) * C[r, :] \
            * np.exp(2j * np.pi * (ell - nu_C).reshape((1,-1)) * x.reshape((-1,1)))
    one_app = (1.-one_app.real)**2 + one_app.imag**2
    map_error = np.sum(one_app,  axis=1)/M
    return map_error


def calc_map_error(gridder, grid_correction, nu, x, W):
    M = len(nu) // W
    C = gridder_to_C(gridder, W)
    return calc_map_error_from_C(C, grid_correction, nu[:M], x, W)


def eskapprox(parm, nu, x, W):
    nunorm=2*nu/W
    beta=parm[0]
    e1 = 0.5 if len(parm)<2 else parm[1]
    e2 = 2. if len(parm)<3 else parm[2]
    return np.exp(beta*W*((1-nunorm**e2)**e1-1))

def getmaxerr(approx, coeff, nu, x, W, M, N, x0):
    nu=(np.arange(W*M)+0.5)/(2*M)
    x=np.arange(N+1)/(2*N)
    krn = approx(coeff, nu, x, W)
    err = kernel2error(krn, nu, x, W)
    err = err[0:int(2*x0*N+0.9999)+1]
    return np.max(np.abs(err))

def scan_esk(rbeta, re0, nu, x, W, M, N, x0, nsamp):
    curmin=1e30
    for e0 in np.linspace(re0[0], re0[1], nsamp):
        for beta in np.linspace(rbeta[0], rbeta[1], nsamp):
            test = getmaxerr(eskapprox, [beta,e0], nu, x, W, M, N, x0)
            if test<curmin:
                curmin, coeffmin = test, [beta,e0]
    return coeffmin

def kernel2error(krn, nu, x, W):
    corr = gridder_to_grid_correction(krn, nu, x, W)
    return calc_map_error(krn, corr, nu, x, W)

M=128
N=512
x=np.arange(N+1)/(2*N)

# for quick experiments, just enter the desired oversampling factor and support
# as single elements in the tuples below
ofactors = np.linspace(1.15,2.00,18)
Ws = np.arange(4,17)
results = []
for W in Ws:
    for ofactor in ofactors:
        x0 = 0.5/ofactor
        nu=(np.arange(W*M)+0.5)/(2*M)
        ulim = int(2*x0*N+0.9999)+1
        rbeta=[1., 2.5]
        re0=[0.48, 0.65]
        dbeta = rbeta[1]-rbeta[0]
        de0 = re0[1]-re0[0]
        for i in range(30):
            res1 = scan_esk(rbeta, re0, nu, x, W, M, N, x0, 10)
            dbeta*=0.5
            de0*=0.5
            rbeta = [res1[0]-0.5*dbeta, res1[0]+0.5*dbeta]
            re0 = [res1[1]-0.5*de0, res1[1]+0.5*de0]
        krn1 = eskapprox(res1, nu, x, W) 
        err1 = kernel2error(krn1, nu, x, W)
        maxerr1 = np.sqrt(np.max(err1[0:ulim]))
        print("{{{0:2d}, {1:4.2f}, {2:13.8g}, {3:12.10f}, {4:12.10f}}},".format(W, ofactor, maxerr1, res1[0], res1[1]))
