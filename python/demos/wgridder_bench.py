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

from time import time

import ducc0.wgridder as wgridder
import matplotlib.pyplot as plt
import numpy as np


def main():
#    ms, fov_deg, npixdirty = '/home/martin/ms/supernovashell.55.7+3.4.spw0.npz', 2., 1200
#    ms, fov_deg, npixdirty = '/home/martin/ms/1052736496-averaged.npz', 25., 2048
    ms, fov_deg, npixdirty = '/home/martin/ms/1052735056.npz', 45., 1200
#    ms, fov_deg, npixdirty = '/home/martin/ms/G330.89-0.36.npz', 2., 1200

    data = np.load(ms)
    uvw, freq, vis, wgt, flags = data["uvw"], data["freqs"], data["vis"], data["wgt"], data["mask"]

    flags[vis==0] = False
    flags[wgt==0] = False
    DEG2RAD = np.pi/180
    pixsize = fov_deg/npixdirty*DEG2RAD
    nthreads = 2
    epsilon = 1e-4
    print('Start gridding...')
    do_wstacking = True

    t0 = time()

    dirty = wgridder.ms2dirty(uvw, freq, vis, wgt, npixdirty, npixdirty, pixsize,
                              pixsize, 0, 0, epsilon, do_wstacking, nthreads, verbosity=1, mask=flags)
    print('Done')
    t = time() - t0
    print("{} s".format(t))
    t0 = time()
    _ = wgridder.dirty2ms(uvw, freq, dirty, wgt, pixsize,
                          pixsize, 0, 0, epsilon, do_wstacking, nthreads, verbosity=1, mask=flags)
    print('Done')
    t = time() - t0
    print("{} s".format(t))
    print("{} visibilities/thread/s".format(np.sum(wgt != 0)/nthreads/t))
    plt.imshow(dirty.T, origin='lower')
    plt.show()


if __name__ == "__main__":
    main()
