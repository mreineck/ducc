import ducc0
import numpy as np
from time import time

try:
    import finufft
    have_finufft = True
except ImportError:
    have_finufft = False

class Bench:
    def __init__(self, shape, npoints):
        self._shape = shape
        ndim = len(shape)
        self._coord = (2*np.pi*np.random.uniform(size=(npoints,ndim)) - np.pi).astype(np.float32)
        # cidx = ((self._coord[:,-1]+np.pi)/(2*np.pi)*1000).astype(np.int64)
        # if ndim > 1:
            # cidx += ((self._coord[:,-2]+np.pi)/(2*np.pi)*1000).astype(np.int64)*1000
        # if ndim > 2:
            # cidx += ((self._coord[:,-3]+np.pi)/(2*np.pi)*1000).astype(np.int64)*1000000
        # idx = np.argsort(cidx)
        # self._coord = self._coord[idx]
        self._points = (np.random.uniform(size=npoints)-0.5
                 + 1j * np.random.uniform(size=npoints)-0.5j).astype(np.complex64)
        self._values = (np.random.uniform(size=shape)-0.5
                 + 1j * np.random.uniform(size=shape)-0.5j).astype(np.complex64)

        self._res_fiducial_1 = ducc0.nufft.nu2u(
            points=self._points.astype(np.complex128),
            coord=self._coord.astype(np.float64),
            forward=True,
            epsilon=3e-13,
            nthreads=0,
            verbosity=1,
            out=np.empty(shape, dtype=np.complex128))

        self._res_fiducial_2 = ducc0.nufft.u2nu(
            grid=self._values.astype(np.complex128),
            coord=self._coord.astype(np.float64),
            forward=True,
            epsilon=3e-13,
            nthreads=0,
            verbosity=1)

    def run(self, epsilon, singleprec, nthreads):
        rdtype = np.float32 if singleprec else np.float64
        dtype = np.complex64 if singleprec else np.complex128

        res={}
        res["shape"] = self._shape
        res["npoints"] = self._coord.shape[0]
        res["epsilon"] = epsilon
        res["nthreads"] = nthreads
        res["singleprec"] = singleprec

        shape = self._shape
        ndim = len(shape)
        npoints = self._coord.shape[0]
        coord = self._coord.astype(rdtype)
        points = self._points.astype(dtype)
        values = self._values.astype(dtype)

        out = np.ones(shape, dtype=dtype)
        plan = ducc0.nufft.plan(gridding=True, coord=coord, grid_shape=shape, epsilon=epsilon, nthreads=nthreads)
        t0 = time()
        res_ducc = plan.nu2u(points=points, forward=True, verbosity=1, out=out)
        res["ducc_trans_1"] = time()-t0
        res["err_ducc_trans_1"] = ducc0.misc.l2error(res_ducc, self._res_fiducial_1)

        out=np.ones(shape=(npoints,), dtype=dtype)
        plan = ducc0.nufft.plan(gridding=False, coord=coord, grid_shape=shape, epsilon=epsilon, nthreads=nthreads)
        t0 = time()
        res_ducc = plan.u2nu(grid=values, forward=True, verbosity=1, out=out)
        res["ducc_trans_2"] = time()-t0
        res["err_ducc_trans_2"] = ducc0.misc.l2error(res_ducc, self._res_fiducial_2)

        out = np.ones(shape, dtype=dtype)
        t0 = time()
        res_ducc = ducc0.nufft.nu2u(points=points, coord=coord, forward=True, epsilon=epsilon, nthreads=nthreads, verbosity=1, out=out)
        res["ducc_full_1"] = time()-t0
        res["err_ducc_1"] = ducc0.misc.l2error(res_ducc, self._res_fiducial_1)

        out=np.ones(shape=(npoints,), dtype=dtype)
        t0 = time()
        res_ducc = ducc0.nufft.u2nu(grid=values, coord=coord, forward=True, epsilon=epsilon, nthreads=nthreads, verbosity=1)
        res["ducc_full_2"] = time()-t0
        res["err_ducc_2"] = ducc0.misc.l2error(res_ducc, self._res_fiducial_2)
        return res

    def run_finufft(self, epsilon, singleprec, nthreads, res):
        import finufft
        rdtype = np.float32 if singleprec else np.float64
        dtype = np.complex64 if singleprec else np.complex128

        shape = self._shape
        ndim = len(shape)
        npoints = self._coord.shape[0]
        coord = self._coord.astype(rdtype)
        points = self._points.astype(dtype)
        values = self._values.astype(dtype)

        plan1 = finufft.Plan(1, self._shape, 1, eps=epsilon, isign=-1,
                             dtype="complex64" if singleprec else "complex128",
                             nthreads=nthreads, debug=1, fftw=0)
        plan1.setpts(*tuple(coord[:,i]for i in range(coord.shape[1])))
        t0 = time()
        res_finufft = plan1.execute(points)
        res["finufft_full_1"] = time()-t0
        res["err_finufft_1"] = ducc0.misc.l2error(res_finufft, self._res_fiducial_1)
        del plan1

        plan2 = finufft.Plan(2, shape, 1, eps=epsilon, isign=-1,
                            dtype="complex64" if singleprec else "complex128",
                            nthreads=nthreads, debug=1, fftw=0)
        plan2.setpts(*tuple(coord[:,i]for i in range(coord.shape[1])))
        t0 = time()
        res_finufft = plan2.execute(values)
        res["finufft_full_2"] = time()-t0
        res["err_finufft_2"] = ducc0.misc.l2error(res_finufft, self._res_fiducial_2)

        return res


def plot(res, fname):
    import matplotlib.pyplot as plt
    have_finufft = "finufft_full_1" in res[0]
    tducc1 = [r["ducc_full_1"] for r in res]
    tducc2 = [r["ducc_full_2"] for r in res]
    tducct1 = [r["ducc_trans_1"] for r in res]
    tducct2 = [r["ducc_trans_2"] for r in res]
    educc1 = np.array([r["err_ducc_1"] for r in res])
    educc2 = np.array([r["err_ducc_2"] for r in res])
    educct1 = np.array([r["err_ducc_trans_1"] for r in res])
    educct2 = np.array([r["err_ducc_trans_2"] for r in res])
    if have_finufft:
        tfinufft1 = [r["finufft_full_1"] for r in res]
        tfinufft2 = [r["finufft_full_2"] for r in res]
        efinufft1 = np.array([r["err_finufft_1"] for r in res])
        efinufft2 = np.array([r["err_finufft_2"] for r in res])
    eps = np.array([r["epsilon"] for r in res])
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(educc1,tducc1,label="ducc unplanned, type 1")
    plt.plot(educc2,tducc2,label="ducc unplanned, type 2")
    plt.plot(educct1,tducct1,label="ducc planned, type 1")
    plt.plot(educct2,tducct2,label="ducc planned, type 2")
    if have_finufft:
        plt.plot(efinufft1,tfinufft1,label="finufft planned, type 1")
        plt.plot(efinufft2,tfinufft2,label="finufft planned, type 2")
    plt.title("shape={}, npoints={}, nthreads={}".format(res[0]["shape"], res[0]["npoints"], res[0]["nthreads"]))
    plt.xlabel("real error")
    plt.ylabel("t[s]")
    plt.legend()
    plt.savefig(fname)
    plt.close()


def runbench(shape, npoints, nthreads, fname, singleprec=False):
    res=[]
    mybench = Bench(shape, npoints)
    epslist = [4e-6, 1e-5, 1e-4, 1e-3, 1e-2] if singleprec else [4e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    for eps in epslist:
        tres = mybench.run(eps, singleprec, nthreads)
        if have_finufft:
            res.append(mybench.run_finufft(eps, singleprec, nthreads, tres))
        else:
            res.append(tres)
    plot(res, fname)


# FINUFFT benchmarks
runbench((   1000000,),  10000000, 1, "1d.png" , False)
runbench((  10000000,), 100000000, 8, "1da.png", False)
runbench(( 1000,1000,),  10000000, 1, "2d.png" , False)
runbench(( 3162,3162,), 100000000, 8, "2da.png", False)
runbench((100,100,100),  10000000, 1, "3d.png" , False)
runbench((216,216,216), 100000000, 8, "3da.png", False)
# NFFT.jl benchmarks
#runbench(( 512*512,),  512*512, 1, "bench_1d.png", False)
#runbench(( 512,512,),  512*512, 1, "bench_2d.png", False)
#runbench((64,64,64,), 64*64*64, 1, "bench_3d.png", False)
