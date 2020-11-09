import numpy as np
import sys


def extra_checks(t):
    if len(set(t.getcol('FIELD_ID'))) != 1:
        raise RuntimeError
    if len(set(t.getcol('DATA_DESC_ID'))) != 1:
        raise RuntimeError


def read_ms(name):
    # Assumptions:
    # - Only one field
    # - Only one spectral window
    # - Flag both LL and RR if one is flagged
    # - Visibilities in DATA column
    from os.path import join
    from casacore.tables import table

    def get_indices(name):
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
            # Can also deal with partially flagged entry
            tflags = np.all(tflags.astype(np.bool), axis=-1)
            twgt = t.getcol(weightcol, startrow=start, nrow=stop-start)[..., ind]
            twgt = np.sum(twgt, axis=-1)
            tflags[twgt == 0] = True

            active_rows[start:stop] = np.invert(np.all(tflags, axis=-1))
            active_channels = np.logical_or(active_channels, np.invert(np.all(tflags, axis=0)))
            start = stop

        nrealrows, nrealchan = np.sum(active_rows), np.sum(active_channels)
        start, realstart = 0, 0
        vis = np.empty((nrealrows, nrealchan), dtype=np.complex64)
        wgtshp = (nrealrows, nrealchan) if fullwgt else (nrealrows,)
        wgt = np.empty(wgtshp, dtype=np.float32)
        while start < nrow:
            stop = min(nrow, start+step)
            realstop = realstart+np.sum(active_rows[start:stop])
            if realstop > realstart:
                allrows = stop-start == realstop-realstart

                twgt = t.getcol(weightcol, startrow=start, nrow=stop-start)[..., ind]
                assert twgt.dtype == np.float32
                if not allrows:
                    twgt = twgt[active_rows[start:stop]]
                if fullwgt:
                    twgt = twgt[:, active_channels]

                tvis = t.getcol("DATA", startrow=start, nrow=stop-start)[..., ind]
                assert tvis.dtype == np.complex64
                if not allrows:
                    tvis = tvis[active_rows[start:stop]]
                tvis = tvis[:, active_channels]

                tflags = t.getcol("FLAG", startrow=start, nrow=stop-start)[..., ind]
                if not allrows:
                    tflags = tflags[active_rows[start:stop]]
                tflags = tflags[:, active_channels]

                # Noise-weighted average
                if not fullwgt:
                    twgt = twgt[:, None]
                assert tflags.dtype == np.bool
                assert twgt.shape[2] == 2
                twgt = twgt*(~tflags)
                tvis = np.sum(twgt*tvis, axis=-1)[..., None]
                twgt = np.sum(twgt, axis=-1)[..., None]
                tvis /= twgt

                vis[realstart:realstop] = tvis
                wgt[realstart:realstop] = twgt
            start, realstart = stop, realstop
        uvw = t.getcol("UVW")[active_rows]

    print('# Rows: {} ({} fully flagged)'.format(nrow, nrow-vis.shape[0]))
    print('# Channels: {} ({} fully flagged)'.format(nchan, nchan-vis.shape[1]))
    print('# Correlations: {}'.format(ncorr))
    print('Full weights' if fullwgt else 'Row-only weights')
    print("{} % flagged".format(np.sum(wgt == 0)/wgt.size*100)
    freq = freq[active_channels]

    # blow up wgt to the right dimensions if necessary
    if not fullwgt:
        wgt = np.broadcast_to(wgt.reshape((-1, 1)), vis.shape)

    uvw = np.ascontiguousarray(uvw)
    freq = np.ascontiguousarray(freq)
    vis = np.ascontiguousarray(vis)
    wgt = np.ascontiguousarray(wgt)
    vis[wgt == 0] = 0.
    return dict(uvw=uvw, freqs=freq, vis=vis, wgt=wgt)


def main():
    if len(sys.argv) != 3:
        raise RuntimeError("bad number of command line arguments")
    datasetname = sys.argv[1]
    outname = sys.argv[2]
    dset = read_ms(datasetname)
    np.savez_compressed(outname, **dset)


if __name__ == '__main__':
    main()
