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


def get_indices(name):
    from os.path import join
    from casacore.tables import table
    with table(join(name, 'POLARIZATION'), readonly=True, ack=False) as t:
        pol = list(t.getcol('CORR_TYPE')[0])
    if set(pol) <= set([5, 6, 7, 8]):
        ind = [pol.index(5), pol.index(8)]
    else:
        ind = [pol.index(9), pol.index(12)]
    return ind


def determine_weighting(t):
    fullwgt = False
    weightcol = "WEIGHT"
    try:
        t.getcol("WEIGHT_SPECTRUM", startrow=0, nrow=1)
        weightcol = "WEIGHT_SPECTRUM"
        fullwgt = True
    except:
        pass
    return fullwgt, weightcol


def extra_checks(t):
    if len(set(t.getcol('FIELD_ID'))) != 1:
        raise RuntimeError
    if len(set(t.getcol('DATA_DESC_ID'))) != 1:
        raise RuntimeError


def read_ms_i(name):
    # Assumptions:
    # - Only one field
    # - Only one spectral window
    # - Flag both LL and RR if one is flagged
    from os.path import join
    from casacore.tables import table

    with table(join(name, 'SPECTRAL_WINDOW'), readonly=True, ack=False) as t:
        freq = t.getcol('CHAN_FREQ')[0]
    nchan = freq.shape[0]
    ind = get_indices(name)
    with table(name, readonly=True, ack=False) as t:
        fullwgt, weightcol = determine_weighting(t)
        extra_checks(t)

        nrow = t.nrows()
        active_rows = np.ones(nrow, dtype=np.bool)
        active_channels = np.zeros(nchan, dtype=np.bool)

        step = max(1, nrow//100)  # how many rows to read in every step

        # determine which subset of rows/channels we need to input
        start = 0
        while start < nrow:
            stop = min(nrow, start+step)
            tflags = t.getcol('FLAG', startrow=start, nrow=stop-start)
            ncorr = tflags.shape[2]
            tflags = tflags[..., ind]
            tflags = np.any(tflags.astype(np.bool), axis=-1)
            twgt = t.getcol(weightcol, startrow=start, nrow=stop-start)[..., ind]
            twgt = 1/np.sum(1/twgt, axis=-1)
            tflags[twgt==0] = True

            active_rows[start:stop] = np.invert(np.all(tflags, axis=-1))
            active_channels = np.logical_or(active_channels, np.invert(np.all(tflags, axis=0)))
            start = stop

        nrealrows, nrealchan = np.sum(active_rows), np.sum(active_channels)
        start, realstart = 0, 0
        vis = np.empty((nrealrows, nrealchan), dtype=np.complex64)
        wgtshp = (nrealrows, nrealchan) if fullwgt else (nrealrows,)
        wgt = np.empty(wgtshp, dtype=np.float32)
        flags = np.empty((nrealrows, nrealchan), dtype=np.bool)
        while start < nrow:
            stop = min(nrow, start+step)
            realstop = realstart+np.sum(active_rows[start:stop])
            if realstop > realstart:
                allrows = stop-start == realstop-realstart
                tvis = t.getcol("DATA", startrow=start, nrow=stop-start)[..., ind]
                tvis = np.sum(tvis, axis=-1)
                if not allrows:
                    tvis = tvis[active_rows[start:stop]]
                tvis = tvis[:, active_channels]
                tflags = t.getcol('FLAG', startrow=start, nrow=stop-start)[..., ind]
                tflags = np.any(tflags.astype(np.bool), axis=-1)
                if not allrows:
                    tflags = tflags[active_rows[start:stop]]
                tflags = tflags[:, active_channels]
                twgt = t.getcol(weightcol, startrow=start, nrow=stop-start)[..., ind]
                twgt = 1/np.sum(1/twgt, axis=-1)
                if not allrows:
                    twgt = twgt[active_rows[start:stop]]
                if fullwgt:
                    twgt = twgt[:, active_channels]
                tflags[twgt==0] = True

                vis[realstart:realstop] = tvis
                wgt[realstart:realstop] = twgt
                flags[realstart:realstop] = tflags

            start, realstart = stop, realstop
        uvw = t.getcol("UVW")[active_rows]

    print('# Rows: {} ({} fully flagged)'.format(nrow, nrow-vis.shape[0]))
    print('# Channels: {} ({} fully flagged)'.format(nchan, nchan-vis.shape[1]))
    print('# Correlations: {}'.format(ncorr))
    print('Full weights' if fullwgt else 'Row-only weights')
    nflagged = np.sum(flags) + (nrow-nrealrows)*nchan + (nchan-nrealchan)*nrow
    print("{} % flagged".format(nflagged/(nrow*nchan)*100))
    freq = freq[active_channels]

    # blow up wgt to the right dimensions if necessary
    if not fullwgt:
        wgt = np.broadcast_to(wgt.reshape((-1,1)), vis.shape)

    return (np.ascontiguousarray(uvw),
            np.ascontiguousarray(freq),
            np.ascontiguousarray(vis),
            np.ascontiguousarray(wgt) if fullwgt else wgt,
            1-flags.astype(np.uint8))


def main():
#    ms, fov_deg = '/home/martin/ms/supernovashell.55.7+3.4.spw0.ms', 2.
#    ms, fov_deg = '/home/martin/ms/1052736496-averaged.ms', 45.
    ms, fov_deg = '/home/martin/ms/1052735056.ms', 45.
#    ms, fov_deg = '/home/martin/ms/cleaned_G330.89-0.36.ms', 2.
    uvw, freq, vis, wgt, flags = read_ms_i(ms)

    npixdirty = 1200
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
