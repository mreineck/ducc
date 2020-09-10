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
# Copyright(C) 2019 Max-Planck-Society

from time import time

import matplotlib.pyplot as plt
import ducc0.wgridder as wgridder
import numpy as np


def read_ms(name):
    # Assumptions:
    # - Only one field
    # - Only one spectral window
    # - Flag both LL and RR if one is flagged
    from os.path import join
    from casacore.tables import table

    with table(join(name, 'SPECTRAL_WINDOW'), readonly=True, ack=False) as t:
        freq = t.getcol('CHAN_FREQ')[0]
    nchan = freq.shape[0]
    with table(join(name, 'POLARIZATION'), readonly=True, ack=False) as t:
        pol = list(t.getcol('CORR_TYPE')[0])
    if set(pol) <= set([5, 6, 7, 8]):
        ind = [pol.index(5), pol.index(8)]
    else:
        ind = [pol.index(9), pol.index(12)]
    with table(name, readonly=True, ack=False) as t:
        if len(set(t.getcol('FIELD_ID'))) != 1:
            raise RuntimeError
        if len(set(t.getcol('DATA_DESC_ID'))) != 1:
            raise RuntimeError
        uvw = t.getcol("UVW")
        nrow = uvw.shape[0]
        step = 1000 # how many rows to read in every step
        start = 0
        vis = np.empty((nrow, nchan), dtype=np.complex64)
        wgt = np.empty((nrow, nchan), dtype=np.float32)
        flags = np.empty((nrow, nchan), dtype=np.bool)
        while start < nrow:
            stop = min(nrow, start+step)
            tvis = t.getcol("DATA", startrow=start, nrow=stop-start)
            ncorr = tvis.shape[2]
            tvis = np.sum(tvis[:, :, ind], axis=2)
            vis[start:stop, :] = tvis
            twgt = t.getcol("WEIGHT", startrow=start, nrow=stop-start)
            twgt = 1/np.sum(1/twgt, axis=1)
            wgt[start:stop, :] = np.repeat(twgt[:, None], len(freq), axis=1)
            tflags = t.getcol('FLAG', startrow=start, nrow=stop-start)
            flags[start:stop, :] = np.any(tflags.astype(np.bool), axis=2)
            start = stop
    # flagged visibilities get weight 0
    wgt[flags] = 0
    # visibilities with weight 0 might as well be flagged
    flags[wgt==0] = True

    print('# Rows: {}'.format(vis.shape[0]))
    print('# Channels: {}'.format(vis.shape[1]))
    print('# Correlations: {}'.format(ncorr))
    print("{} % flagged".format(np.sum(flags)/flags.size*100))

    # cut out unused rows/channels
    rows_with_data = np.invert(np.all(flags, axis=1))
    n_empty_rows = nrow-np.sum(rows_with_data)
    print("Completely flagged rows: {}".format(n_empty_rows))
    if n_empty_rows > 0:
        uvw = uvw[rows_with_data,:]
        vis = vis[rows_with_data,:]
        wgt = wgt[rows_with_data,:]
        flags = flags[rows_with_data,:]
    channels_with_data = np.invert(np.all(flags, axis=0))
    n_empty_channels = nchan-np.sum(channels_with_data)
    print("Completely flagged channels: {}".format(n_empty_channels))
    if n_empty_channels > 0:
        freq = freq[channels_with_data]
        vis = vis[:, channels_with_data]
        wgt = wgt[:, channels_with_data]
        flags = flags[:, channels_with_data]
    return (np.ascontiguousarray(uvw),
            np.ascontiguousarray(freq),
            np.ascontiguousarray(vis),
            np.ascontiguousarray(wgt),
            1-flags.astype(np.uint8))


def main():
    ms = '/home/martin/ms/supernovashell.55.7+3.4.spw0.ms'
    ms = '/home/martin/ms/1052735056_cleaned.ms'
    uvw, freq, vis, wgt, flags = read_ms(ms)

    npixdirty = 1200
    DEG2RAD = np.pi/180
    pixsize = 45/npixdirty*DEG2RAD
    nthreads = 2
    epsilon = 1e-4
    print('Start gridding...')
    do_wstacking = True

    wgt = np.where(np.abs(vis)==0, 0, wgt)
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
