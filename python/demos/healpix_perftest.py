import time
import numpy as np
import ducc0.healpix as ph

rng = np.random.default_rng(42)


def report(name, vlen, ntry, nside, isnest, perf):
    print(name, ": ", perf*1e-6, "MOps/s", sep="")


def random_ptg(vlen):
    res = np.empty((vlen, 2), dtype=np.float64)
    res[:, 0] = np.arccos((rng.random(vlen)-0.5)*2)
    res[:, 1] = rng.random(vlen)*2*np.pi
    return res


def random_pix(nside, vlen):
    return rng.integers(low=0, high=12*nside*nside-1, size=vlen, dtype=np.int64)


def dummy(vlen):
    _ = np.zeros(vlen, dtype=np.int64)


def genperf(func, fname, inp, vlen, ntry, nside, isnest):
    cnt = 0
    t = time.time()
    while cnt < ntry:
        func(inp)
        cnt += 1
    t = time.time()-t
    p = (vlen*ntry)/t
    report(fname, vlen, ntry, nside, isnest, p)


def perf_pix2ang(vlen, ntry, nside, isnest):
    inp = random_pix(nside, vlen)
    base = ph.Healpix_Base(nside, "NEST" if isnest else "RING")
    genperf(base.pix2ang, "pix2ang", inp, vlen, ntry, nside, isnest)


def perf_ang2pix(vlen, ntry, nside, isnest):
    inp = random_ptg(vlen)
    base = ph.Healpix_Base(nside, "NEST" if isnest else "RING")
    genperf(base.ang2pix, "ang2pix", inp, vlen, ntry, nside, isnest)


def perf_pix2vec(vlen, ntry, nside, isnest):
    inp = random_pix(nside, vlen)
    base = ph.Healpix_Base(nside, "NEST" if isnest else "RING")
    genperf(base.pix2vec, "pix2vec", inp, vlen, ntry, nside, isnest)


def perf_vec2pix(vlen, ntry, nside, isnest):
    inp = ph.ang2vec(random_ptg(vlen))
    base = ph.Healpix_Base(nside, "NEST" if isnest else "RING")
    genperf(base.vec2pix, "vec2pix", inp, vlen, ntry, nside, isnest)


def perf_ring2nest(vlen, ntry, nside, isnest):
    inp = random_pix(nside, vlen)
    base = ph.Healpix_Base(nside, "NEST" if isnest else "RING")
    genperf(base.ring2nest, "ring2nest", inp, vlen, ntry, nside, isnest)


def perf_nest2ring(vlen, ntry, nside, isnest):
    inp = random_pix(nside, vlen)
    base = ph.Healpix_Base(nside, "NEST" if isnest else "RING")
    genperf(base.nest2ring, "nest2ring", inp, vlen, ntry, nside, isnest)


def perf_neighbors(vlen, ntry, nside, isnest):
    inp = random_pix(nside, vlen)
    base = ph.Healpix_Base(nside, "NEST" if isnest else "RING")
    genperf(base.neighbors, "neighbors", inp, vlen, ntry, nside, isnest)


def suite(vlen, ntry, nside, isnest):
    print("vlen=", vlen, ", ", "NEST" if isnest else "RING", sep="")
    dummy(vlen)
    perf_pix2ang(vlen, ntry, nside, isnest)
    perf_ang2pix(vlen, ntry, nside, isnest)
    perf_pix2vec(vlen, ntry, nside, isnest)
    perf_vec2pix(vlen, ntry, nside, isnest)
    perf_neighbors(vlen, ntry, nside, isnest)


nside = 512
ntry = 1000
print("nside=", nside, sep="")
for vlen in (1, 10, 100, 1000, 10000):
    for isnest in (True, False):
        suite(vlen, ntry, nside, isnest)
        perf_ring2nest(vlen, ntry, nside, isnest)
        perf_nest2ring(vlen, ntry, nside, isnest)
        print()
