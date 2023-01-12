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
# Copyright(C) 2019-2020 Max-Planck-Society

import os
from time import time

import ducc0.wgridder.experimental as wgridder
import ducc0
import matplotlib.pyplot as plt
import numpy as np


def get_npixdirty(uvw, freq, fov_deg, mask):
    speedOfLight = 299792458.
    bl = np.sqrt(uvw[:,0]**2+uvw[:,1]**2+uvw[:,2]**2)
    bluvw = bl.reshape((-1,1))*freq.reshape((1,-1))/speedOfLight
    maxbluvw = np.max(bluvw*mask)
    minsize = int((2*fov_deg*np.pi/180*maxbluvw)) + 1
    return minsize+(minsize%2)  # make even


def load_ms(ms):
    import resolve as rve
    ms = next(rve.ms2observations_all("/data/CYG-ALL-13360-8MHZ.ms", "DATA"))
    return dict(uvw=ms.uvw, freqs=ms.freq, vis=ms.vis.val[0], wgt=ms.weight.val_rw()[0],
                mask=ms.mask.val_rw()[0].astype(np.uint8))


def main():
#    ms, fov_deg = '/home/martin/ms/supernovashell.55.7+3.4.spw0.npz', 2.
#    ms, fov_deg = '/home/martin/ms/1052736496-averaged.npz', 25.
#    ms, fov_deg = '/home/martin/ms/1052735056.npz', 45.
    ms, fov_deg = '/home/martin/ms/G330.89-0.36.npz', 2.
#    ms, fov_deg = '/home/martin/ms/bigms.npz', 0.0005556*1800
#    ms, fov_deg = '/home/martin/ms/L_UV_DATA-IF1.npz', 1.
#    ms, fov_deg = '/data/CYG-ALL-13360-8MHZ.npz', 0.08
#    ms, fov_deg = '/data/L_UV_DATA-IF1.npz', 1.
#    ms, fov_deg = '/home/martin/ms/ska_low.npz', 5.5*2  # for 16384 px

    if os.path.splitext(ms)[1] == ".ms":
        data = load_ms(ms)
    else:
        data = np.load(ms)
    uvw, freq, vis, wgt = data["uvw"], data["freqs"], data["vis"], data["wgt"]
    mask = data["mask"] if "mask" in data else None
    wgt[vis == 0] = 0
    if mask is None:
        mask = np.ones(wgt.shape, dtype=np.uint8)
    mask[wgt == 0] = False
    DEG2RAD = np.pi/180
    nthreads = 8
    epsilon = 1e-4
    do_wgridding = True
    verbosity = 1

    do_sycl = False #True
    do_cng = False #True

    ntries = 1

    npixdirty = get_npixdirty(uvw, freq, fov_deg, mask)
    pixsize = fov_deg/npixdirty*DEG2RAD

  #  vis = vis.astype(np.complex128)
  #  wgt = wgt.astype(np.float64)

    print('CPU gridding...')
    print()
    mintime=1e300
    for _ in range(ntries):
        t0 = time()
        dirty = wgridder.vis2dirty(
            uvw=uvw, freq=freq, vis=vis, wgt=wgt,
            mask=mask, npix_x=npixdirty, npix_y=npixdirty, pixsize_x=pixsize,
            pixsize_y=pixsize, epsilon=epsilon, do_wgridding=do_wgridding,
            nthreads=nthreads, verbosity=verbosity, flip_v=False,
            double_precision_accumulation=False)
        mintime = min(mintime, time()-t0)
    print()
    print("Best time: {:.4f} s".format(mintime))
    print("{:.2f} Mvis/s".format(np.sum(wgt != 0)/mintime/1e6))
    print()

    print('Tuned CPU gridding...')
    print()
    mintime=1e300
    for _ in range(ntries):
        t0 = time()
        dirty_t = wgridder.vis2dirty_tuning(
            uvw=uvw, freq=freq, vis=vis, wgt=wgt,
            mask=mask, npix_x=npixdirty, npix_y=npixdirty, pixsize_x=pixsize,
            pixsize_y=pixsize, epsilon=epsilon, do_wgridding=do_wgridding,
            nthreads=nthreads, verbosity=verbosity, flip_v=False,
            double_precision_accumulation=False)
        mintime = min(mintime, time()-t0)
    print()
    print("Best time: {:.4f} s".format(mintime))
    print("{:.2f} Mvis/s".format(np.sum(wgt != 0)/mintime/1e6))
    print("L2 error compared to untuned CPU: {:.2e}".format(ducc0.misc.l2error(dirty,dirty_t)))
    print()

#    import matplotlib.pyplot as plt
#    plt.imshow(dirty)
#    plt.show()

    if do_sycl:
        print('SYCL gridding...')
        mintime=1e300
        for _ in range(ntries):
            t0 = time()
            dirty_g = wgridder.vis2dirty(
                uvw=uvw, freq=freq, vis=vis, wgt=wgt,
                mask=mask, npix_x=npixdirty, npix_y=npixdirty, pixsize_x=pixsize,
                pixsize_y=pixsize, epsilon=epsilon, do_wgridding=do_wgridding,
                nthreads=nthreads, verbosity=verbosity, flip_v=False, gpu=True,
                double_precision_accumulation=False)
            mintime = min(mintime, time()-t0)
        print("Best time: {:.4f} s".format(mintime))
        print("{:.2f} Mvis/s".format(np.sum(wgt != 0)/mintime/1e6))
        print("L2 error compared to CPU: {:.2e}".format(ducc0.misc.l2error(dirty,dirty_g)))
        print()
    if do_cng:
        import cuda_nifty_gridder as cng
        print('ska-gridder-nifty-cuda gridding...')
        print()
        mintime=1e300
        for _ in range(ntries):
            t0 = time()
            dirty_cng = cng.ms2dirty(uvw, freq, vis, wgt, npixdirty, npixdirty,
              pixsize, pixsize, 0, 0, epsilon, do_wgridding, verbosity=verbosity)
            mintime = min(mintime, time()-t0)
        print()
        print("Best time: {:.4f} s".format(mintime))
        print("{:.2f} Mvis/s".format(np.sum(wgt != 0)/mintime/1e6))
        print("L2 error compared to CPU: {:.2e}".format(ducc0.misc.l2error(dirty,dirty_cng)))
        print()

    vis_out = vis.copy()
    print('CPU degridding...')
    print()
    mintime=1e300
    for _ in range(ntries):
        t0 = time()
        vis_out = wgridder.dirty2vis(
            uvw=uvw, freq=freq, dirty=dirty, wgt=wgt,
            mask=mask, pixsize_x=pixsize, pixsize_y=pixsize, epsilon=epsilon,
            do_wgridding=do_wgridding, nthreads=nthreads, verbosity=verbosity,
            flip_v=False, vis=vis_out)
        mintime = min(mintime, time()-t0)
    print()
    print("Best time: {:.4f} s".format(mintime))
    print("{:.2f} Mvis/s".format(np.sum(wgt != 0)/mintime/1e6))
    print()
    vis_out_t = vis.copy()
    print('Tuned CPU degridding...')
    print()
    mintime=1e300
    for _ in range(ntries):
        t0 = time()
        vis_out_t = wgridder.dirty2vis_tuning(
            uvw=uvw, freq=freq, dirty=dirty, wgt=wgt,
            mask=mask, pixsize_x=pixsize, pixsize_y=pixsize, epsilon=epsilon,
            do_wgridding=do_wgridding, nthreads=nthreads, verbosity=verbosity,
            flip_v=False, vis=vis_out_t)
        mintime = min(mintime, time()-t0)
    print()
    print("Best time: {:.4f} s".format(mintime))
    print("{:.2f} Mvis/s".format(np.sum(wgt != 0)/mintime/1e6))
    print("L2 error compared to untuned CPU: {:.2e}".format(ducc0.misc.l2error(vis_out,vis_out_t)))
    print()
    del vis_out_t
    if do_sycl:
        vis_out_g = vis.copy()
        print('SYCL degridding...')
        print()
        mintime=1e300
        for _ in range(ntries):
            t0 = time()
            vis_out_g = wgridder.dirty2vis(
                uvw=uvw, freq=freq, dirty=dirty, wgt=wgt,
                mask=mask, pixsize_x=pixsize, pixsize_y=pixsize, epsilon=epsilon,
                do_wgridding=do_wgridding,
                nthreads=nthreads, verbosity=verbosity,
                flip_v=False,
                gpu=True, vis=vis_out_g)
            mintime = min(mintime, time()-t0)
        print()
        print("Best time: {:.4f} s".format(mintime))
        print("{:.2f} Mvis/s".format(np.sum(wgt != 0)/mintime/1e6))
        print("L2 error compared to CPU: {:.2e}".format(ducc0.misc.l2error(vis_out,vis_out_g)))
        print()
        del vis_out_g
    if do_cng:
        import cuda_nifty_gridder as cng
        print('ska-gridder-nifty-cuda degridding...')
        print()
        mintime=1e300
        for _ in range(ntries):
            t0 = time()
            vis_out_cng = cng.dirty2ms(uvw, freq, dirty, wgt, pixsize, pixsize, 0, 0, epsilon, do_wgridding, verbosity=verbosity)
            mintime = min(mintime, time()-t0)
            vis_out_cng *= wgt
        print()
        print("Best time: {:.4f} s".format(mintime))
        print("{:.2f} Mvis/s".format(np.sum(wgt != 0)/mintime/1e6))
        print("L2 error compared to CPU: {:.2e}".format(ducc0.misc.l2error(vis_out,vis_out_cng)))
        print()
        del vis_out_cng


if __name__ == "__main__":
    main()
