# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2022-2025 Max-Planck-Society

import ducc0
import numpy as np
from time import time

try:
    import finufft
    have_finufft = True
except ImportError:
    have_finufft = False

class Bench12:
    def __init__(self, shape, npoints):
        self._shape = shape
        ndim = len(shape)
        # We create the random values in single precision, just to make sure
        # we don't change them by truncating in case we run a single precision
        # transform.

        # random nonuniform coordinates in [-pi; pi]
        self._coord = (2*np.pi*np.random.uniform(size=(npoints,ndim)) - np.pi).astype(np.float32)
        # random nonuniform strengths with -0.5 < re, im < 0.5
        self._points = (np.random.uniform(size=npoints)-0.5
                 + 1j * np.random.uniform(size=npoints)-0.5j).astype(np.complex64)
        # random uniform grid values -0.5 < re, im < 0.5
        self._values = (np.random.uniform(size=shape)-0.5
                 + 1j * np.random.uniform(size=shape)-0.5j).astype(np.complex64)

        # Produce "ground truth", i.e. run NUFFT with the best available
        # precision
        print("computing reference results with high precision ...")
        eps = 1.0001*ducc0.nufft.bestEpsilon(ndim=ndim, singleprec=False)
        self._res_fiducial_1 = ducc0.nufft.nu2u(
            points=self._points.astype(np.complex128),
            coord=self._coord.astype(np.float64),
            forward=True,
            epsilon=eps,
            nthreads=0,
            verbosity=0,
            out=np.empty(shape, dtype=np.complex128))

        self._res_fiducial_2 = ducc0.nufft.u2nu(
            grid=self._values.astype(np.complex128),
            coord=self._coord.astype(np.float64),
            forward=True,
            epsilon=eps,
            nthreads=0,
            verbosity=0)
        print("done")

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
        t0 = time()
        plan = ducc0.nufft.plan(nu2u=True, coord=coord, grid_shape=shape, epsilon=epsilon, nthreads=nthreads)
        res["ducc_1_planned_time_plan"] = time()-t0
        t0 = time()
        out = plan.nu2u(points=points, forward=True, verbosity=0, out=out)
        res["ducc_1_planned_time_exec"] = time()-t0
        res["ducc_1_planned_err"] = ducc0.misc.l2error(out, self._res_fiducial_1)
        print(f"ducc0,     planned, type 1: time={res["ducc_1_planned_time_exec"]}, L2 error={res["ducc_1_planned_err"]}")
        del plan, out

        out=np.ones(shape=(npoints,), dtype=dtype)
        t0 = time()
        plan = ducc0.nufft.plan(nu2u=False, coord=coord, grid_shape=shape, epsilon=epsilon, nthreads=nthreads)
        res["ducc_2_planned_time_plan"] = time()-t0
        t0 = time()
        out = plan.u2nu(grid=values, forward=True, verbosity=0, out=out)
        res["ducc_2_planned_time_exec"] = time()-t0
        res["ducc_2_planned_err"] = ducc0.misc.l2error(out, self._res_fiducial_2)
        print(f"ducc0,     planned, type 2: time={res["ducc_2_planned_time_exec"]}, L2 error={res["ducc_2_planned_err"]}")
        del plan, out

        out = np.ones(shape, dtype=dtype)
        t0 = time()
        out = ducc0.nufft.nu2u(points=points, coord=coord, forward=True, epsilon=epsilon, nthreads=nthreads, verbosity=0, out=out)
        res["ducc_1_unplanned_time_full"] = time()-t0
        res["ducc_1_unplanned_err"] = ducc0.misc.l2error(out, self._res_fiducial_1)
        print(f"ducc0,   unplanned, type 1: time={res["ducc_1_unplanned_time_full"]}, L2 error={res["ducc_1_unplanned_err"]}")
        del out

        out=np.ones(shape=(npoints,), dtype=dtype)
        t0 = time()
        out = ducc0.nufft.u2nu(grid=values, coord=coord, forward=True, epsilon=epsilon, nthreads=nthreads, verbosity=0, out=out)
        res["ducc_2_unplanned_time_full"] = time()-t0
        res["ducc_2_unplanned_err"] = ducc0.misc.l2error(out, self._res_fiducial_2)
        print(f"ducc0,   unplanned, type 2: time={res["ducc_2_unplanned_time_full"]}, L2 error={res["ducc_2_unplanned_err"]}")
        del out

        if not have_finufft:
            return res

        import finufft
        coord = tuple(np.ascontiguousarray(coord[:,i]) for i in range(coord.shape[1]))
        func1=[finufft.nufft1d1, finufft.nufft2d1, finufft.nufft3d1]
        func2=[finufft.nufft1d2, finufft.nufft2d2, finufft.nufft3d2]

        t0 = time()
        plan = finufft.Plan(1, self._shape, 1, eps=epsilon, isign=-1,
                            dtype="complex64" if singleprec else "complex128",
                            nthreads=nthreads, debug=0)
        plan.setpts(*coord)
        res["finufft_1_planned_time_plan"] = time()-t0
        out = np.ones(shape, dtype=dtype)
        t0 = time()
        out = plan.execute(points, out=out)
        res["finufft_1_planned_time_exec"] = time()-t0
        res["finufft_1_planned_err"] = ducc0.misc.l2error(out, self._res_fiducial_1)
        print(f"Finufft,   planned, type 1: time={res["finufft_1_planned_time_exec"]}, L2 error={res["finufft_1_planned_err"]}")
        del plan, out

        t0 = time()
        plan = finufft.Plan(2, shape, 1, eps=epsilon, isign=-1,
                            dtype="complex64" if singleprec else "complex128",
                            nthreads=nthreads, debug=0)
        plan.setpts(*coord)
        res["finufft_2_planned_time_plan"] = time()-t0
        out = np.ones((npoints,), dtype=dtype)
        t0 = time()
        out = plan.execute(values, out=out)
        res["finufft_2_planned_time_exec"] = time()-t0
        res["finufft_2_planned_err"] = ducc0.misc.l2error(out, self._res_fiducial_2)
        print(f"Finufft,   planned, type 2: time={res["finufft_2_planned_time_exec"]}, L2 error={res["finufft_2_planned_err"]}")
        del plan, out

        out = np.ones(shape, dtype=dtype)
        t0=time()
        out = func1[ndim-1](*coord, points, out=out, eps=epsilon, isign=-1, nthreads=nthreads, debug=0)
        res["finufft_1_unplanned_time_exec"] = time()-t0
        res["finufft_1_unplanned_err"] = ducc0.misc.l2error(out, self._res_fiducial_1)
        print(f"Finufft, unplanned, type 1: time={res["finufft_1_unplanned_time_exec"]}, L2 error={res["finufft_1_unplanned_err"]}")
        del out

        out = np.ones((npoints,), dtype=dtype)
        t0=time()
        out = func2[ndim-1](*coord, values, out=out, eps=epsilon, isign=-1, nthreads=nthreads, debug=0)
        res["finufft_2_unplanned_time_exec"] = time()-t0
        res["finufft_2_unplanned_err"] = ducc0.misc.l2error(out, self._res_fiducial_2)
        print(f"Finufft, unplanned, type 2: time={res["finufft_2_unplanned_time_exec"]}, L2 error={res["finufft_2_unplanned_err"]}")
        del out

        return res


class Bench3:
    def __init__(self, npoints_in, npoints_out, minmax_in, minmax_out):
        self._npoints_in = npoints_in
        self._npoints_out = npoints_out
        self._minmax_in = minmax_in
        self._minmax_out = minmax_out
        ndim = minmax_in.shape[1]
        # We create the random values in single precision, just to make sure
        # we don't change them by truncating in case we run a single precision
        # transform.

        # random nonuniform input coordinates
        self._coord_in = np.random.uniform(size=(npoints_in,ndim)).astype(np.float32)
        self._coord_in *= minmax_in[1]-minmax_in[0]
        self._coord_in += minmax_in[0]
        # random nonuniform input coordinates
        self._coord_out = np.random.uniform(size=(npoints_out,ndim)).astype(np.float32)
        self._coord_out *= minmax_out[1]-minmax_out[0]
        self._coord_out += minmax_out[0]
        # random nonuniform strengths with -0.5 < re, im < 0.5
        self._points = (np.random.uniform(size=npoints_in)-0.5
                 + 1j * np.random.uniform(size=npoints_in)-0.5j).astype(np.complex64)

        # Produce "ground truth", i.e. run NUFFT with the best available
        # precision
        print("computing reference results with high precision ...")
        eps = 1.0001*ducc0.nufft.bestEpsilon(ndim=ndim, singleprec=False)
        self._res_fiducial = ducc0.nufft.experimental.nu2nu(
            points_in=self._points.astype(np.complex128),
            coord_in=self._coord_in.astype(np.float64),
            coord_out=self._coord_out.astype(np.float64),
            forward=True, epsilon=eps, verbosity=0, nthreads=0)
        print("done")

    def run(self, epsilon, singleprec, nthreads):
        rdtype = np.float32 if singleprec else np.float64
        dtype = np.complex64 if singleprec else np.complex128

        res={}
        res["npoints_in"] = self._coord_in.shape[0]
        res["npoints_out"] = self._coord_out.shape[0]
        res["epsilon"] = epsilon
        res["nthreads"] = nthreads
        res["singleprec"] = singleprec

        ndim = self._minmax_in.shape[1]
        npoints_in = self._coord_in.shape[0]
        coord_in = self._coord_in.astype(rdtype)
        npoints_out = self._coord_out.shape[0]
        coord_out = self._coord_out.astype(rdtype)
        points = self._points.astype(dtype)

        out = np.ones((npoints_out,), dtype=dtype)
        t0 = time()
        plan = ducc0.nufft.experimental.plan3(coord_in=coord_in,
           coord_out=coord_out, epsilon=epsilon, verbosity=0, nthreads=nthreads)
        res["ducc_3_planned_time_plan"] = time()-t0
        t0 = time()
        out = plan.exec(forward=True,points_in=points,points_out=out)
        res["ducc_3_planned_time_exec"] = time()-t0
        res["ducc_3_planned_err"] = ducc0.misc.l2error(out, self._res_fiducial)
        print(f"ducc0,     planned, type 3: time={res["ducc_3_planned_time_exec"]}, L2 error={res["ducc_3_planned_err"]}")
        del plan, out

        out = np.ones((npoints_in,), dtype=dtype)
        t0 = time()
        out = ducc0.nufft.experimental.nu2nu(points_in=points, coord_in=coord_in, coord_out=coord_out, forward=True, epsilon=epsilon, verbosity=0, nthreads=nthreads, points_out=out)
        res["ducc_3_unplanned_time_full"] = time()-t0
        res["ducc_3_unplanned_err"] = ducc0.misc.l2error(out, self._res_fiducial)
        print(f"ducc0,   unplanned, type 3: time={res["ducc_3_unplanned_time_full"]}, L2 error={res["ducc_3_unplanned_err"]}")
        del out

        if not have_finufft:
            return res

        import finufft
        coord_in = tuple(np.ascontiguousarray(coord_in[:,i]) for i in range(coord_in.shape[1]))
        coord_out = tuple(np.ascontiguousarray(coord_out[:,i]) for i in range(coord_out.shape[1]))

        out = np.ones((npoints_out,), dtype=dtype)
        plan = finufft.Plan(3, ndim, eps=epsilon, isign=-1, dtype=dtype, nthreads=nthreads)
        args = list(coord_in) + [None]*(3-ndim) + list(coord_out)
        plan.setpts(*args)
        t0 = time()
        out = plan.execute(points, out=out)
        res["finufft_3_planned_time_exec"] = time()-t0
        res["finufft_3_planned_err"] = ducc0.misc.l2error(out, self._res_fiducial)
        print(f"finufft,   planned, type 3: time={res["finufft_3_planned_time_exec"]}, L2 error={res["finufft_3_planned_err"]}")
        del plan, out


        out = np.ones((npoints_out,), dtype=dtype)
        func=[finufft.nufft1d3, finufft.nufft2d3, finufft.nufft3d3]
        t0 = time()
        out = func[ndim-1](*coord_in, points, *coord_out, out=out, eps=epsilon, isign=-1, nthreads=nthreads)
        res["finufft_3_unplanned_time_full"] = time()-t0
        res["finufft_3_unplanned_err"] = ducc0.misc.l2error(out, self._res_fiducial)
        print(f"finufft, unplanned, type 3: time={res["finufft_3_unplanned_time_full"]}, L2 error={res["finufft_3_unplanned_err"]}")
        del out

        return res


def plot(res, fname):
    import matplotlib.pyplot as plt
    fct = 1e9/res[0]["npoints"]
    have_finufft = "finufft_1_planned_time_exec" in res[0]
    tducc1 = fct*np.array([r["ducc_1_unplanned_time_full"] for r in res])
    tducc2 = fct*np.array([r["ducc_2_unplanned_time_full"] for r in res])
    tducct1 = fct*np.array([r["ducc_1_planned_time_exec"] for r in res])
    tducct2 = fct*np.array([r["ducc_2_planned_time_exec"] for r in res])
    educc1 = np.array([r["ducc_1_unplanned_err"] for r in res])
    educc2 = np.array([r["ducc_2_unplanned_err"] for r in res])
    educct1 = np.array([r["ducc_1_planned_err"] for r in res])
    educct2 = np.array([r["ducc_2_planned_err"] for r in res])
    if have_finufft:
        tfinufftt1 = fct*np.array([r["finufft_1_planned_time_exec"] for r in res])
        tfinufftt2 = fct*np.array([r["finufft_2_planned_time_exec"] for r in res])
        tfinufft1 = fct*np.array([r["finufft_1_unplanned_time_exec"] for r in res])
        tfinufft2 = fct*np.array([r["finufft_2_unplanned_time_exec"] for r in res])
        efinufftt1 = np.array([r["finufft_1_planned_err"] for r in res])
        efinufftt2 = np.array([r["finufft_2_planned_err"] for r in res])
        efinufft1 = np.array([r["finufft_1_unplanned_err"] for r in res])
        efinufft2 = np.array([r["finufft_2_unplanned_err"] for r in res])
    eps = np.array([r["epsilon"] for r in res])
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(educc1,tducc1,label="ducc unplanned, type 1")
    plt.plot(educc2,tducc2,label="ducc unplanned, type 2")
    plt.plot(educct1,tducct1,label="ducc planned, type 1")
    plt.plot(educct2,tducct2,label="ducc planned, type 2")
    if have_finufft:
        plt.plot(efinufftt1,tfinufftt1,label="finufft planned, type 1")
        plt.plot(efinufftt2,tfinufftt2,label="finufft planned, type 2")
        plt.plot(efinufft1,tfinufft1,label="finufft unplanned, type 1")
        plt.plot(efinufft2,tfinufft2,label="finufft unplanned, type 2")
    plt.title("shape={}, npoints={}, nthreads={}".format(res[0]["shape"], res[0]["npoints"], res[0]["nthreads"]))
    plt.xlabel("real error")
    plt.ylabel("ns per nonuniform point")
    plt.legend()
    plt.savefig(fname, bbox_inches='tight')
    plt.close()


def plot3(res, fname):
    import matplotlib.pyplot as plt
    fct = 1e9/(res[0]["npoints_in"]+ res[0]["npoints_out"])
    have_finufft = "finufft_3_planned_time_exec" in res[0]
    tducc3 = fct*np.array([r["ducc_3_unplanned_time_full"] for r in res])
    tducct3 = fct*np.array([r["ducc_3_planned_time_exec"] for r in res])
    educc3 = np.array([r["ducc_3_unplanned_err"] for r in res])
    educct3 = np.array([r["ducc_3_planned_err"] for r in res])
    if have_finufft:
        tfinufftt3 = fct*np.array([r["finufft_3_planned_time_exec"] for r in res])
        tfinufft3 = fct*np.array([r["finufft_3_unplanned_time_full"] for r in res])
        efinufftt3 = np.array([r["finufft_3_planned_err"] for r in res])
        efinufft3 = np.array([r["finufft_3_unplanned_err"] for r in res])
    eps = np.array([r["epsilon"] for r in res])
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(educc3,tducc3,label="ducc unplanned, type 3")
    plt.plot(educct3,tducct3,label="ducc planned, type 3")
    if have_finufft:
        plt.plot(efinufft3,tfinufft3,label="finufft unplanned, type 3")
        plt.plot(efinufftt3,tfinufftt3,label="finufft planned, type 3")
    plt.title("npoints_in={}, npoint_out={}, nthreads={}".format(res[0]["npoints_in"], res[0]["npoints_out"], res[0]["nthreads"]))
    plt.xlabel("real error")
    plt.ylabel("ns per nonuniform point")
    plt.legend()
    plt.savefig(fname, bbox_inches='tight')
    plt.close()


def runbench12(shape, npoints, nthreads, fname, singleprec=False):
    res=[]
    mybench = Bench12(shape, npoints)
    if singleprec:
        epslist = [[2.5e-7, 4.5e-7, 8.2e-7][len(shape)-1], 1e-5, 1e-4, 1e-3, 1e-2]
    else:
        epslist = [[4e-15, 8e-15, 2e-14][len(shape)-1], 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    for eps in epslist:
        print(f"N={shape}, M={npoints}, epsilon={eps}, nthreads={nthreads}:")
        res.append(mybench.run(eps, singleprec, nthreads))
        print()
    plot(res, fname)

def runbench3(npoints_in, npoints_out, minmax_in, minmax_out, nthreads, fname, singleprec=False):
    res=[]
    mybench = Bench3(npoints_in, npoints_out, minmax_in, minmax_out)
    ndim = minmax_in.shape[1]
    if singleprec:
        epslist = [[2.5e-7, 4.5e-7, 8.2e-7][ndim-1], 1e-5, 1e-4, 1e-3, 1e-2]
    else:
        epslist = [[4e-15, 8e-15, 2e-14][ndim-1], 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    for eps in epslist:
        print(f"{ndim}D, M_in={npoints_in}, M_out={npoints_out}, epsilon={eps}, nthreads={nthreads}:")
        res.append(mybench.run(eps, singleprec, nthreads))
        print()
    plot3(res, fname)


singleprec = False
# FINUFFT benchmarks, analogous to figures 6.1-6.3 in the 2018 paper
if True:
    runbench12((   1000000,),  10000000, 1, "finufft_1d_serial.png"  , singleprec)
    runbench12(( 1000,1000,),  10000000, 1, "finufft_2d_serial.png"  , singleprec)
    runbench12((100,100,100),  10000000, 1, "finufft_3d_serial.png"  , singleprec)
    runbench12((  10000000,), 100000000, 8, "finufft_1d_parallel.png", singleprec)
    runbench12(( 3162,3162,), 100000000, 8, "finufft_2d_parallel.png", singleprec)
    runbench12((216,216,216), 100000000, 8, "finufft_3d_parallel.png", singleprec)
# NFFT.jl benchmarks, lower nonuniform point density
if True:
    runbench12(( 512*512,),  512*512, 1, "bench_1d.png", singleprec)
    runbench12(( 512,512,),  512*512, 1, "bench_2d.png", singleprec)
    runbench12((64,64,64,), 64*64*64, 1, "bench_3d.png", singleprec)
# some preliminary type 3 benchmarks
if True:
    # helper function to create coordinate ranges fpr NU points
    def make_ranges(xmin, xmax, ymin=None, ymax=None, zmin=None, zmax=None):
        if ymin is None:
            return np.array([[xmin], [xmax]])
        if zmin is None:
            return np.array([[xmin, ymin], [xmax, ymax]])
        return np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
    minmax = make_ranges(-1., 1.)
    runbench3(1000000,  1000000, minmax, minmax, 1, "finufft_1d_type3_serial.png", singleprec)
    runbench3(10000000,  10000000, minmax, minmax, 8, "finufft_1d_type3_parallel.png"  , singleprec)
    minmax = make_ranges(-1., 1., -1., 1.)
    runbench3(1000000,  1000000, minmax, minmax, 1, "finufft_2d_type3_serial.png", singleprec)
    runbench3(10000000,  10000000, minmax, minmax, 8, "finufft_2d_type3_parallel.png"  , singleprec)
    minmax = make_ranges(-1., 1., -1., 1., -1., 1.)
    runbench3(1000000,  1000000, minmax, minmax, 1, "finufft_3d_type3_serial.png", singleprec)
    runbench3(10000000,  10000000, minmax, minmax, 8, "finufft_3d_type3_parallel.png"  , singleprec)
