import numpy as np
import ducc0

def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def random_alm(lmax, mmax, spin, ncomp, rng):
    res = rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax))) \
     + 1j*rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax)))
    # make a_lm with m==0 real-valued
    res[:, 0:lmax+1].imag = 0.
    ofs=0
    for s in range(spin):
        res[:, ofs:ofs+spin-s] = 0.
        ofs += lmax+1-s
    return res

def alm_lval(lmax, mmax):
    res = np.empty(nalm(lmax,mmax), dtype=np.int64)
    ofs=0
    rng = np.arange(lmax+1)
    for m in range(mmax+1):
        res[ofs+m:ofs+lmax+1] = rng[m:]
        ofs += lmax-m
    return res

def alm_mval(lmax, mmax):
    res = np.empty(nalm(lmax,mmax), dtype=np.int64)
    ofs=0
    for m in range(mmax+1):
        res[ofs+m:ofs+lmax+1] = m
        ofs += lmax-m
    return res

# converts from shtns spherical/toroidal coefficients to ducc0 spin1 ones
def sphtor_to_spin1(alm, lmax, mmax):
    el = alm_lval(lmax, mmax)
    res = alm*np.sqrt(el*(el+1))
    res[1] *= -1
    return res


# converts from ducc0 spin1 coefficients to shtns spherical/toroidal ones
def spin1_to_sphtor(alm, lmax, mmax):
    el = alm_lval(lmax, mmax)
    tmp = np.sqrt(el*(el+1))
    tmp[0] = 1
    res = alm/tmp
    res[1] *= -1
    return res

def rec_coeffs(lmax, mmax):
    el = alm_lval(lmax, mmax)
    em = alm_mval(lmax, mmax)
    el[0] = 1  # warning fix
    res = np.sqrt((el+em)*(el-em)/((2*el+1)*(2*el-1)))
    res[0] = 0
    return res

def mul_costheta_matrix_shifted(lmax, mmax):
    res = np.zeros(2*nalm(lmax,mmax))
    res[0:-2:2] = res[1:-1:2] = rec_coeffs(lmax, mmax)[1:]
    return res

def stdt_matrix_shifted(lmax, mmax):
    res = mul_costheta_matrix_shifted(lmax, mmax)
    el = alm_lval(lmax, mmax)
    res[::2] *= -(el+2)
    res[1::2] *= el
    return res

def apply_stdt_matrix(alm, lmax, mmax):
    res = np.zeros((nalm(lmax+1,mmax),), dtype=np.complex128)
    stdt = stdt_matrix_shifted(lmax+1, mmax)
    ofs = ofs2 = 0
    for m in range(mmax+1):
        # contribution from l-1
        res[ofs2+m+1:ofs2+lmax+2] += alm[ofs+m:ofs+lmax+1]*stdt[2*(ofs2+m)+1:2*(ofs2+lmax+1)+1:2]
        # contribution from l+1
        res[ofs2+m:ofs2+lmax] += alm[ofs+m+1:ofs+lmax+1]*stdt[2*(ofs2+m):2*(ofs2+lmax):2]
        ofs += lmax-m
        ofs2 += lmax+1-m
    return res

def apply_stdt_matrix_backwards(alm, lmax, mmax):
    res = np.zeros((nalm(lmax,mmax),), dtype=np.complex128)
    stdt = stdt_matrix_shifted(lmax, mmax)
    ofs = 0
    for m in range(mmax+1):
        # contribution from l-1
        res[ofs+m+1:ofs+lmax+1] += alm[ofs+m:ofs+lmax]*stdt[2*(ofs+m):2*(ofs+lmax):2]
        # contribution from l+1
        res[ofs+m:ofs+lmax] += alm[ofs+m+1:ofs+lmax+1]*stdt[2*(ofs+m)+1:2*(ofs+lmax)+1:2]
        ofs += lmax-m
    return res

def apply_ddphi_matrix(alm, lmax, mmax):
    return alm*1j*alm_mval(lmax, mmax)

def increase_lmax_by_n(alm, lmax, mmax, n):
    res = np.zeros((nalm(lmax+n,mmax),), dtype=np.complex128)
    ofs = ofs2 = 0
    for m in range(mmax+1):
        res[ofs2+m:ofs2+lmax+1] = alm[ofs+m:ofs+lmax+1]
        ofs += lmax-m
        ofs2 += lmax+n-m
    return res
def decrease_lmax_by_n(alm, lmax, mmax, n):
    res = np.zeros((nalm(lmax-n,mmax),), dtype=np.complex128)
    ofs = ofs2 = 0
    for m in range(mmax+1):
        res[ofs2+m:ofs2+lmax-n+1] = alm[ofs+m:ofs+lmax-n+1]
        ofs += lmax-m
        ofs2 += lmax-n-m
    return res

def spin1to0(alm, lmax, mmax):
    alm = spin1_to_sphtor(alm,lmax, mmax)
    phipart0 = increase_lmax_by_n(-apply_ddphi_matrix(alm[1],lmax,mmax),lmax,mmax,1)
    phipart1 = increase_lmax_by_n(-apply_ddphi_matrix(alm[0],lmax,mmax),lmax,mmax,1)
    thetapart0 = -apply_stdt_matrix(alm[0],lmax,mmax)
    thetapart1 =  apply_stdt_matrix(alm[1],lmax,mmax)
    return np.vstack([phipart0+thetapart0, phipart1+thetapart1])

def spin0to1(alm, lmax, mmax):
    phipart0 = -apply_ddphi_matrix(alm[1],lmax,mmax)
    phipart1 = -apply_ddphi_matrix(alm[0],lmax,mmax)
    thetapart0 =  apply_stdt_matrix_backwards(alm[0],lmax,mmax)
    thetapart1 = -apply_stdt_matrix_backwards(alm[1],lmax,mmax)
    res = np.vstack([phipart0+thetapart0, phipart1+thetapart1])
    el = alm_lval(lmax, mmax)
    el[0] = 1  # warning fix
    res /= (el*(el+1)).reshape((1,-1))
    res[:,0] = 0
    return sphtor_to_spin1(res, lmax, mmax)

def synthesis_spin1_via_spin0(alm, lmax, mmax, ntheta, nphi):
    almx = spin1to0(alm, lmax, mmax)
    res = np.vstack([ducc0.sht.synthesis_2d(alm=ax.reshape((1,-1)), lmax=lmax+1, mmax=mmax, spin=0, geometry="GL", ntheta=ntheta, nphi=nphi) for ax in almx])
    sintheta = np.sin(ducc0.misc.GL_thetas(ntheta)).reshape((1,-1,1))
    res /= -sintheta
    return res

def analysis_spin1_via_spin0(map, lmax, mmax):
    ntheta, nphi = map.shape[1:]
    sintheta = np.sin(ducc0.misc.GL_thetas(ntheta))
    wgt = ducc0.misc.GL_weights(ntheta, nphi)
    map2 = map*(wgt/sintheta).reshape((1,-1,1))
    res = np.vstack([ducc0.sht.adjoint_synthesis_2d(map=mp.reshape((1,ntheta,nphi)), lmax=lmax+1, mmax=mmax, spin=0, geometry="GL") for mp in map2])
    res = spin0to1(res,lmax+1,mmax)
    res = np.array([decrease_lmax_by_n(r,lmax+1,mmax,1) for r in res])
    return res

def compare_spin1 (lmax, mmax):
    print("lmax =",lmax, "mmax =", mmax)
    rng = np.random.default_rng(42)
    alm_ref = random_alm(lmax, mmax, 1, 2, rng)

    ntheta, nphi= lmax+1, 2*mmax+1
    map_ref = ducc0.sht.synthesis_2d(alm=alm_ref, lmax=lmax, mmax=mmax, spin=1, geometry="GL", ntheta=ntheta, nphi=nphi)

    map_test = synthesis_spin1_via_spin0(alm_ref, lmax, mmax, ntheta, nphi)
    print(ducc0.misc.l2error(map_ref, map_test))

    alm_test = analysis_spin1_via_spin0(map_ref, lmax, mmax)
    print(ducc0.misc.l2error(alm_ref,alm_test))


for i in range(20):
    lmax = np.random.randint(2,5000)
    mmax = np.random.randint(2,lmax)
    compare_spin1(lmax, mmax)
