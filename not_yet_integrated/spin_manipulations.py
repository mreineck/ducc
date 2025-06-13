import numpy as np
import shtns
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


def shtns_alm_rec(lmax, mmax):
    res = np.zeros((nalm(lmax,mmax),2))
    el = alm_lval(lmax, mmax)
    em = alm_mval(lmax, mmax)

    res[:,0] = -np.sqrt((2*el+1)/(2*el-3) * (el-1+em)*(el-1-em)/((el+em)*(el-em)))
    res[:,1] = np.sqrt((2*el+1)*(2*el-1)/((el+em)*(el-em)))
    return res


def mul_costheta_matrix_shifted(lmax, mmax):
    res = np.zeros(2*nalm(lmax,mmax))
    res[0:-2:2] = res[1:-1:2] = (1./shtns_alm_rec(lmax, mmax)[1:,1])
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
        res[ofs2+m+1:ofs2+lmax+2] += alm[ofs+m:ofs+lmax+1]*stdt[2*(ofs2+m+1)-1:2*(ofs2+lmax+2)-1:2]
        # contribution from l+1
        res[ofs2+m:ofs2+lmax] += alm[ofs+m+1:ofs+lmax+1]*stdt[2*(ofs2+m):2*(ofs2+lmax):2]
        ofs += lmax-m
        ofs2 += lmax+1-m
    return res

def apply_stdt_matrix_backwards(alm, lmax, mmax):
    res = np.zeros((nalm(lmax+1,mmax),), dtype=np.complex128)
    stdt = stdt_matrix_shifted(lmax+1, mmax)
    ofs = ofs2 = 0
    for m in range(mmax+1):
#Sl[l] = mx(l,m,1)*v(l+1,m) + mx(l-1,m,0)*v(l-1,m) - i*m*w(l,m)
#Tl[l] = -mx(l,m,1)*w(l+1,m) -mx(l-1,m,0)*w(l-1,m) - i*m*v(l,m)
        # contribution from l-1
        res[ofs2+m+1:ofs2+lmax+1] += alm[ofs+m:ofs+lmax]*stdt[2*(ofs2+m):2*(ofs2+lmax):2]
        # contribution from l+1
        res[ofs2+m:ofs2+lmax] += alm[ofs+m+1:ofs+lmax+1]*stdt[2*(ofs2+m)+1:2*(ofs2+lmax)+1:2]
        ofs += lmax-m
        ofs2 += lmax+1-m
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

#/// Vlm =  st*d(Slm)/dtheta + I*m*Tlm
#/// Wlm = -st*d(Tlm)/dtheta + I*m*Slm
def spin1to0(alm, lmax, mmax):
    alm = spin1_to_sphtor(alm,lmax, mmax)
    phipart0 = increase_lmax_by_n(-apply_ddphi_matrix(alm[1],lmax,mmax),lmax,mmax,1)
    phipart1 = increase_lmax_by_n(-apply_ddphi_matrix(alm[0],lmax,mmax),lmax,mmax,1)
    thetapart0 = -apply_stdt_matrix(alm[0],lmax,mmax)
    thetapart1 =  apply_stdt_matrix(alm[1],lmax,mmax)
    return np.vstack([phipart0+thetapart0, phipart1+thetapart1])

#/// Slm = - (I*m*Wlm + MX*Vlm) / (l*(l+1))		=> why does this work ??? (aliasing of 1/sin(theta) ???)
#/// Tlm = - (I*m*Vlm - MX*Wlm) / (l*(l+1))
def spin0to1(alm, lmax, mmax):
    phipart0 = increase_lmax_by_n(-apply_ddphi_matrix(alm[1],lmax,mmax),lmax,mmax,1)
    phipart1 = increase_lmax_by_n(-apply_ddphi_matrix(alm[0],lmax,mmax),lmax,mmax,1)
    thetapart0 =  apply_stdt_matrix_backwards(alm[0],lmax,mmax)
    thetapart1 = -apply_stdt_matrix_backwards(alm[1],lmax,mmax)
    res = np.vstack([phipart0+thetapart0, phipart1+thetapart1])
    el = alm_lval(lmax+1, mmax)
    res /= (el*(el+1)).reshape((1,-1))
    res[:,0] = 0
    return sphtor_to_spin1(res, lmax+1, mmax)

def compare_spin1 (lmax):
    print("lmax=",lmax)
    mmax=lmax
    rng = np.random.default_rng(42)
    alm = random_alm(lmax, lmax, 1, 2, rng)

#    print(alm)
    # alm2 = spin1to0(alm, lmax, mmax)
    # alm3 = spin0to1(alm2, lmax+1, mmax)
    # # #almx = increase_lmax_by_1(increase_lmax_by_1(alm,lmax,mmax),lmax+1,mmax)
    # print(alm)
    # print(alm3)
    # print(alm3[:,0:5]/alm[:,0:5])
    # exit()

    ntheta, nphi= lmax+2, 2*lmax+1

    map_ref = ducc0.sht.synthesis_2d(alm=alm, lmax=lmax, mmax=mmax, spin=1, geometry="GL", ntheta=ntheta, nphi=nphi)

    almx = spin1to0(alm, lmax, mmax)
    map2=[ducc0.sht.synthesis_2d(alm=ax.reshape((1,-1)), lmax=lmax+1, mmax=mmax, spin=0, geometry="GL", ntheta=ntheta, nphi=nphi)[0] for ax in almx]
    sintheta = np.sin(ducc0.misc.GL_thetas(ntheta)).reshape((-1,1))
    map2[0] /= -sintheta
    map2[1] /= -sintheta
    for mr, m2 in zip(map_ref, map2):
        print(ducc0.misc.l2error(mr, m2))

    map2[0] /= sintheta
    map2[1] /= sintheta
    alm2=[ducc0.sht.analysis_2d(map=mp.reshape((1,ntheta,nphi)), lmax=lmax+1, mmax=mmax, spin=0, geometry="GL")[0] for mp in map2]
    almy = spin0to1(alm2,lmax+1,mmax)
    almy = np.array([decrease_lmax_by_n(ay,lmax+2,mmax,2) for ay in almy])
  #  print(alm)
  #  print(almy)
    print(ducc0.misc.l2error(alm,almy))
  #  exit()
    # map3 = ducc0.sht.synthesis_2d(alm=almy, lmax=lmax+2, mmax=mmax, spin=1, geometry="GL", ntheta=ntheta, nphi=nphi)
    # for mr, m3 in zip(map_ref, map3):
        # print(ducc0.misc.l2error(mr, m3))

for i in range(20):
    compare_spin1(lmax=np.random.randint(2,5000))
#compare_spin1(lmax=2047)
