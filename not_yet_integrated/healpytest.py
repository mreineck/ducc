import healpy as hp
import numpy as np

def _sym_ortho(a, b):
    if b == 0:
        return np.sign(a), 0, abs(a)
    elif a == 0:
        return 0, np.sign(b), abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = np.sign(b) / np.sqrt(1+tau*tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = np.sign(a) / np.sqrt(1+tau*tau)
        s = c * tau
        r = a / c
    return c, s, r


def my_lsmr(op, op_dagger, b, n, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8,
         maxiter=None, show=False, x0=None):

    b = np.atleast_1d(b)
    if b.ndim > 1:
        b = b.squeeze()

    msg = ('The exact solution is x = 0, or x = x0, if x0 was given  ',
           'Ax - b is small enough, given atol, btol                  ',
           'The least-squares solution is good enough, given atol     ',
           'The estimate of cond(Abar) has exceeded conlim            ',
           'Ax - b is small enough for this machine                   ',
           'The least-squares solution is good enough for this machine',
           'Cond(Abar) seems to be too large for this machine         ',
           'The iteration limit has been reached                      ')

    hdg1 = '   itn      x(1)       norm r    norm Ar'
    hdg2 = ' compatible   LS      norm A   cond A'
    pfreq = 20   # print frequency (for repeating the heading)
    pcount = 0   # print counter

    m = len(b)
    if x0 is not None:
        if len(x0) != n:
            raise RuntimeError("n mismatch") 

    minDim = min([m, n])

    if maxiter is None:
        maxiter = minDim

    if x0 is None:
        dtype = np.result_type(b, float)
    else:
        dtype = np.result_type(b, x0, float)

    if show:
        print(' ')
        print('LSMR            Least-squares solution of  Ax = b\n')
        print(f'The matrix A has {m} rows and {n} columns')
        print('damp = %20.14e\n' % (damp))
        print('atol = %8.2e                 conlim = %8.2e\n' % (atol, conlim))
        print('btol = %8.2e             maxiter = %8g\n' % (btol, maxiter))

    u = b
    normb = np.linalg.norm(b)
    if x0 is None:
        x = zeros(n, dtype)
        beta = normb.copy()
    else:
        x = np.atleast_1d(x0.copy())
        u = u - op(x)
        beta = np.linalg.norm(u)

    if beta > 0:
        u = (1 / beta) * u
        v = op_dagger(u)
        alpha = np.linalg.norm(v)
    else:
        v = zeros(n, dtype)
        alpha = 0

    if alpha > 0:
        v *= 1 / alpha

    # Initialize variables for 1st iteration.
    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1
    rhobar = 1
    cbar = 1
    sbar = 0

    h = v.copy()
    hbar = np.zeros(h.shape, dtype)

    # Initialize variables for estimation of ||r||.
    betadd = beta
    betad = 0
    rhodold = 1
    tautildeold = 0
    thetatilde = 0
    zeta = 0
    d = 0

    # Initialize variables for estimation of ||A|| and cond(A)
    normA2 = alpha * alpha
    maxrbar = 0
    minrbar = 1e+100
    normA = np.sqrt(normA2)
    condA = 1
    normx = 0

    # Items for use in stopping rules, normb set earlier
    istop = 0
    ctol = 1 / conlim if conlim > 0 else 0
    normr = beta

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    normar = alpha * beta
    if normar == 0:
        if show:
            print(msg[0])
        return x, istop, itn, normr, normar, normA, condA, normx

    if normb == 0:
        x = b
        return x, istop, itn, normr, normar, normA, condA, normx

    if show:
        print(' ')
        print(hdg1, hdg2)
        test1 = 1
        test2 = alpha / beta
        str1 = '%6g %12.5e' % (itn, x[0])
        str2 = ' %10.3e %10.3e' % (normr, normar)
        str3 = '  %8.1e %8.1e' % (test1, test2)
        print(''.join([str1, str2, str3]))

    # Main iteration loop.
    while itn < maxiter:
        itn = itn + 1

        # Perform the next step of the bidiagonalization to obtain the
        # next  beta, u, alpha, v.  These satisfy the relations
        #         beta*u  =  A@v   -  alpha*u,
        #        alpha*v  =  A'@u  -  beta*v.

        u *= -alpha
        u += op(v)
        beta = np.linalg.norm(u)

        if beta > 0:
            u *= (1 / beta)
            v *= -beta
            v += op_dagger(u)
            alpha = np.linalg.norm(v)
            if alpha > 0:
                v *= (1 / alpha)

        # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

        # Construct rotation Qhat_{k,2k+1}.
        chat, shat, alphahat = _sym_ortho(alphabar, damp)

        # Use a plane rotation (Q_i) to turn B_i to R_i
        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s*alpha
        alphabar = c*alpha

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = - sbar * zetabar

        # Update h, h_hat, x.
        hbar *= - (thetabar * rho / (rhoold * rhobarold))
        hbar += h
        x += (zeta / (rho * rhobar)) * hbar
        h *= - (thetanew / rho)
        h += v

        # Estimate of ||r||.

        # Apply rotation Qhat_{k,2k+1}.
        betaacute = chat * betadd
        betacheck = -shat * betadd

        # Apply rotation Q_{k,k+1}.
        betahat = c * betaacute
        betadd = -s * betaacute

        # Apply rotation Qtilde_{k-1}.
        # betad = betad_{k-1} here.

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = - stildeold * betad + ctildeold * betahat

        # betad   = betad_k here.
        # rhodold = rhod_k  here.

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = np.sqrt(d + (betad - taud)**2 + betadd * betadd)

        # Estimate ||A||.
        normA2 = normA2 + beta * beta
        normA = np.sqrt(normA2)
        normA2 = normA2 + alpha * alpha

        # Estimate cond(A).
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        # Test for convergence.

        # Compute norms for convergence testing.
        normar = abs(zetabar)
        normx = np.linalg.norm(x)

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.

        test1 = normr / normb
        test2 = infty if (normA*normr) == 0 else normar / (normA * normr)
        test3 = 1 / condA
        t1 = test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb

        # The following tests guard against extremely small values of
        # atol, btol or ctol.  (The user may have set any or all of
        # the parameters atol, btol, conlim  to 0.)
        # The effect is equivalent to the normAl tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.

        if itn >= maxiter:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        # Allow for tolerances set by the user.
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        # See if it is time to print something.
        if show:
            if (n <= 40) or (itn <= 10) or (itn >= maxiter - 10) or \
               (itn % 10 == 0) or (test3 <= 1.1 * ctol) or \
               (test2 <= 1.1 * atol) or (test1 <= 1.1 * rtol) or \
               (istop != 0):

                if pcount >= pfreq:
                    pcount = 0
                    print(' ')
                    print(hdg1, hdg2)
                pcount = pcount + 1
                str1 = '%6g %12.5e' % (itn, x[0])
                str2 = ' %10.3e %10.3e' % (normr, normar)
                str3 = '  %8.1e %8.1e' % (test1, test2)
                str4 = ' %8.1e %8.1e' % (normA, condA)
                print(''.join([str1, str2, str3, str4]))

        if istop > 0:
            break

    # Print the stopping condition.
    if show:
        print(' ')
        print('LSMR finished')
        print(msg[istop])
        print('istop =%8g    normr =%8.1e' % (istop, normr))
        print('    normA =%8.1e    normAr =%8.1e' % (normA, normar))
        print('itn   =%8g    condA =%8.1e' % (itn, condA))
        print('    normx =%8.1e' % (normx))
        print(str1, str2)
        print(str3, str4)

    return x, istop, itn, normr, normar, normA, condA, normx


def hp_map_analysis_lsq(map, lmax, mmax, tol=1e-10, maxiter=20):
    from scipy.sparse.linalg import LinearOperator, lsqr, lsmr
    nside = hp.npix2nside(map.shape[0])

    # helper functions to convert between real- and complex-valued a_lm
    def alm2realalm(alm):
        res = np.zeros(len(alm)*2-lmax-1)
        res[0:lmax+1] = alm[0:lmax+1].real
        res[lmax+1:] = alm[lmax+1:].view(np.float64)*np.sqrt(2.)
        return res
    def realalm2alm(alm):
        res = np.zeros((len(alm)+lmax+1)//2, dtype=np.complex128)
        res[0:lmax+1] = alm[0:lmax+1]
        res[lmax+1:] = alm[lmax+1:].view(np.complex128)*(np.sqrt(2.)/2)
        return res
   
    def a2m2(x):
        talm = realalm2alm(x)
        return hp.alm2map(talm, lmax=lmax, nside=nside)
    def m2a2(x):
        talm = hp.map2alm(x, lmax=lmax, iter=0)*((12*nside**2)/(4*np.pi))
        return alm2realalm(talm)

    #initial guess
    alm0 = m2a2(map)/len(map)*(4*np.pi)
    op = LinearOperator(matvec=a2m2, rmatvec=m2a2, shape=(len(map),len(alm0)))
 #   res = lsqr(A=op, b=map, n=len(alm0), x0=alm0, atol=tol, btol=tol, iter_lim=maxiter, show=True)
    res = my_lsmr(a2m2, m2a2, n=len(alm0), b=map, x0=alm0, atol=tol, btol=tol, maxiter=maxiter, show=True)
    res = lsmr(A=op, b=map, x0=alm0, atol=tol, btol=tol, maxiter=maxiter, show=True)
    return realalm2alm(res[0])

np.random.seed(42)
nside = 1024
noise_amplitude = 1e0
lmax = int(2*nside)
iter = 50


print(f'# nside = {nside}')
print(f'# lmax = {lmax}')


# generate random a_lm
alm = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
alm.real = np.random.randn(alm.size)
alm.imag[lmax+1:] = np.random.randn(alm.size-lmax-1)

# map is alm2map(alm) plus some white noise
map = hp.alm2map(alm, nside)
map += noise_amplitude*np.random.normal(size=len(map))

# extract a_lm from map using the new algorithm
alm2_new = hp_map_analysis_lsq(map, lmax, lmax, tol=1e-12, maxiter=iter)
map2_new = hp.alm2map(alm2_new, lmax=lmax, nside=nside)
print("relative residual: ",np.sqrt(np.vdot(map-map2_new,map-map2_new)/np.vdot(map,map)))

# extract a_lm from map using the healpy method
alm2_old = hp.map2alm(map, lmax=lmax, iter=iter)
map2_old = hp.alm2map(alm2_old, lmax=lmax, nside=nside)
print("relative residual: ",np.sqrt(np.vdot(map-map2_old,map-map2_old)/np.vdot(map,map)))

