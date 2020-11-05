import numpy as np
import sys

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
#                tflags[twgt==0] = True

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

    vis[wgt==0] = 0.

    return (np.ascontiguousarray(uvw),
            np.ascontiguousarray(freq),
            np.ascontiguousarray(vis),
            np.ascontiguousarray(wgt) if fullwgt else wgt,
            1-flags.astype(np.uint8))


def read_ms(name):
    tmp = read_ms_i(name)
    return dict(uvw=tmp[0],
            freqs=tmp[1],
            vis=tmp[2],
            wgt=tmp[3],
            mask=tmp[4])


def main():
    if len(sys.argv) != 3:
        raise RuntimeError("bad number of command line arguments")
    datasetname = sys.argv[1]
    outname = sys.argv[2]
    dset = read_ms(datasetname)
    np.savez_compressed(outname, **dset)


if __name__ == '__main__':
    main()
