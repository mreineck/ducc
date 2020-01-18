import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=16)

def trap(vec, dx):
    # Perform trapezoidal integration
    return dx * (np.sum(vec) - 0.5 * (vec[0] + vec[-1]))

def make_evaluation_grids(W, M, N):
    """Generate vectors nu and x on which the gridder and gridding correction functions need to be evaluated.
        W is the number of integer gridpoints in total
        M determines the sampling of the nu grid, dnu = 1/(2*M)
        N determines the sampling of the x grid, dx = 1/(2*N)
    """
    nu = (np.arange(W * M, dtype=float) + 0.5) / (2 * M)
    x = np.arange(N+1, dtype=float)/(2 * N)
    return nu, x



def gridder_to_C(gridder, W):
    """Reformat gridder evaluated on the nu grid returned by make_evaluation_grids into the sampled C function
    which has an index for the closest gridpoint and an index for the fractional distance from that gridpoint
    """
    M = len(gridder) // W
    C = np.zeros((W, M), dtype=float)
    for r in range(0, W):
        l = r - (W/2) + 1
        indx = (np.arange(M) - 2 * M * l).astype(int)
        # Use symmetry to deal with negative indices
        indx[indx<0] = -indx[indx<0] - 1
        C[r, :] = gridder[indx]
    return C

def gridder_to_grid_correction(gridder, nu, x, W):
    """Calculate the optimal grid correction function from the gridding function. The vectors x and nu should
    have been constructed using make_evaluation_grids"""
    M = len(nu) // W
    N = len(x) - 1
    dnu = nu[1] - nu[0]
    C = gridder_to_C(gridder, W)
    c = np.zeros(x.shape, dtype=float)
    d = np.zeros(x.shape, dtype=float)
    for n, x_val in enumerate(x):
        for rp in range(0, W):
            lp = rp - (W/2) + 1
            for r in range(0, W):
                l = r - (W/2) + 1
                d[n] += np.sum(C[rp, :] * C[r, :] * np.cos(2 * np.pi * (lp - l) * x_val)) * dnu
            c[n] += np.sum(C[rp, :] * np.cos(2 * np.pi * (lp - nu[:M]) * x_val)) * dnu
    return c/d

def calc_map_error(gridder, grid_correction, nu, x, W):
    M = len(nu) // W
    N = len(x) - 1
    dnu = nu[1] - nu[0]
    C = gridder_to_C(gridder, W)
    loss = np.zeros((len(x), 2, M), dtype=float)
    for n, x_val in enumerate(x):
        one_app = 0
        for r in range(0, W):
            l = r - (W/2) + 1
            one_app += grid_correction[n] * C[r, :] * np.exp(2j * np.pi * (l - nu[:M]) * x_val)
        loss[n, 0, :] = 1.0 - np.real(one_app)
        loss[n, 1, :] = np.imag(one_app)
    map_error = np.zeros(len(x), dtype=float)
    for i in range(len(x)):
        map_error[i] = 2 * np.sum((loss[i, :, :].flatten())**2) * dnu
    return map_error


minsupp=2
maxsupp=9
minx=25
maxx=43
stepx=3

nsteps = 100
maperr = np.zeros(nsteps)
interr = np.zeros(nsteps)
res={}
xval=np.arange(minx, maxx, stepx)
for w in range(minsupp, maxsupp):
    betamin, betamax = 1.2, 2.5
    dbeta = 0.01
    betas = np.arange(betamin, betamax, dbeta)
    nsteps = betas.size
    for i in range(nsteps):
        beta0=betas[i]
        M = 64
        N = 100
        nu, x = make_evaluation_grids(w, M, N)
        beta=beta0*w
        gridfunc = lambda nu: np.exp(beta*(np.sqrt(1.-nu*nu)-1))
        gridder = gridfunc(nu/(w*0.5))
        grid_correction = gridder_to_grid_correction(gridder, nu, x, w)

        map_err = calc_map_error(gridder, grid_correction, nu, x, w)
        for xv in xval:
            x0 = xv*0.01
            which = np.digitize(x0, x)
            maperr= np.max(np.abs(map_err[:which]))
            interr=trap(map_err[:which], 1.0/(2*N))
            if (w, xv) not in res:
                res[(w, xv)] = (maperr, interr, beta0)
                print(w, xv, res[(w, xv)])
            else:
                if interr < res[(w, xv)][1]:
                    res[(w, xv)] = (maperr, interr, beta0)
                    print(w, xv, res[(w, xv)])

def model_interr0(x0,W):
    c1 = -5.7
    return np.exp(c1*W*(1-x0))
def model_interr1(x0,W):
    osf = 1./(x0*2)
    c1 = -2.1
    return np.exp(c1*W*osf)
def model_maxerr0(x0,W):
    osf = 1./(x0*2)
    c1 = -2.0
    return 12*np.exp(c1*W*osf)
def model_beta0(x0,W):
    betacorr=[0,0,-0.51,-0.21,-0.1,-0.05,-0.025,-0.0125,0,0,0,0,0,0,0,0,]
    bcstrength=1.+(x0-0.25)*2.5
    return 2.32+bcstrength*betacorr[W]+(0.25-x0)*3.1

supps=np.arange(minsupp, maxsupp)
for i in range(minx, maxx, stepx):
    #betas = np.array([res[(w,i)][2] for w in range(minsupp, maxsupp)])
    #plt.plot(supps, betas,label="{}".format(i))
    #betas2 = np.array([model_beta0(0.01*i, w) for w in range(minsupp, maxsupp)])
    #plt.plot(supps, betas2,label="m{}".format(i))
    errs = np.array([res[(w,i)][0] for w in range(minsupp, maxsupp)])
#    errs = np.array([res[(w,i)][1] for w in range(minsupp, maxsupp)])
    errs2 = np.array([model_maxerr0(0.01*i, w) for w in range(minsupp, maxsupp)])
    plt.semilogy(supps, errs,label="{}".format(i))
    plt.semilogy(supps, errs2,label="m{}".format(i))
plt.legend()
plt.show()


#log(eps**2) propto W
