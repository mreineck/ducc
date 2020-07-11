import numpy as np
from scipy.optimize import leastsq

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
#            tmp = C[rp, :, :] * C[r, :, :] * np.cos(2 * np.pi * (ellp - ell) * x)
#            print(np.max(np.abs(tmp-np.mean(tmp2,axis=0))))
            d += tmp
#            d += np.mean(tmp,axis=0)
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


import matplotlib.pyplot as plt

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
 #   print(coeff, np.max(err))
    return np.max(np.abs(err))

def scan_esk(rbeta, re0, nu, x, W, M, N, x0):
    curmin=1e30
    for e0 in np.linspace(re0[0], re0[1], 10):
        for beta in np.linspace(rbeta[0], rbeta[1], 10):
            test = getmaxerr(eskapprox, [beta,e0], nu, x, W, M, N, x0)
            if test<curmin:
                curmin, coeffmin = test, [beta,e0]
#                print(coeffmin, np.sqrt(curmin))
    return coeffmin

def kernel2error(krn, nu, x, W):
    corr = gridder_to_grid_correction(krn, nu, x, W)
    return calc_map_error(krn, corr, nu, x, W)

M=128
N=512
x=np.arange(N+1)/(2*N)

# for quick experiments, just enter the desired oversampling factor and support
# as single elements in the tuples below
ofactors = np.linspace(1.2,2.0,9)
Ws = reversed((15,))
results = []
#plt.ion()
np.set_printoptions(floatmode='unique')
for W in Ws:
    for ofactor in ofactors:
        x0 = 0.5/ofactor
#        plt.title("W={}, ofactor={}".format(W, ofactor))
        nu=(np.arange(W*M)+0.5)/(2*M)
        ulim = int(2*x0*N+0.9999)+1
#         res1 = leastsq(lambda c:geterr(eskapprox, c,nu,x,W,M, N, x0, 1), [2.3,0.5], full_output=True)[0]
#         krn1 = eskapprox(res1, nu, x, W) 
#         err1 = kernel2error(krn1, nu, x, W)
#         results.append(err1)
#         maxerr1 = np.sqrt(np.max(err1[0:ulim]))
#         print(W, ofactor, maxerr1, res1)
        rbeta=[1., 2.5]
        re0=[0.48, 0.58]
        dbeta = rbeta[1]-rbeta[0]
        de0 = re0[1]-re0[0]
        for i in range(8):
            res1 = scan_esk(rbeta, re0, nu, x, W, M, N, x0)
            dbeta*=0.25
            de0*=0.25
            rbeta = [res1[0]-0.5*dbeta, res1[0]+0.5*dbeta]
            re0 = [res1[1]-0.5*de0, res1[1]+0.5*de0]
  #      res1 = leastsq(lambda c:geterr(eskapprox, c,nu,x,W,M, N, x0, 1), res1, full_output=True)[0]
        krn1 = eskapprox(res1, nu, x, W) 
#        kpoly,cf = polyize(krn1,W,W+3)
#        print(_l2error(kpoly,krn1))
#        plt.plot(np.log10(np.abs(kpoly-krn1)))
        err1 = kernel2error(krn1, nu, x, W)
#        results.append(err1)
        maxerr1 = np.sqrt(np.max(err1[0:ulim]))
        print("{{{}, {}, {}, {}, {}}},".format(W, ofactor, maxerr1, res1[0], res1[1]))
#        for r in results:
#            plt.semilogy(x, np.sqrt(r))
#        plt.show()
#         print("2XXXX:", maxerr1, res1)
#         plt.semilogy(x, np.sqrt(err1), label="ESK (2 params)")
#         res1 = leastsq(lambda c:geterr(eskapprox, c,nu,x,W,M, N, x0), [res1[0], res1[1], 2.], full_output=True)[0]
#         krn1 = eskapprox(res1, nu, x, W) 
#         err1 = kernel2error(krn1, nu, x, W)
#         maxerr1 = np.sqrt(np.max(err1[0:ulim]))
# #         print("3XXXX:", maxerr1, res1)
#         plt.semilogy(x, np.sqrt(err1), label="ESK (2 params)")
#         plt.axvline(x=x0)
#         plt.axhline(y=maxerr1)
#         plt.legend()
#         plt.show()

