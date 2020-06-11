import ducc0.healpix as ph
import numpy as np

rng = np.random.default_rng(42)


def random_ptg(vlen):
    res = np.empty((vlen, 2), dtype=np.float64)
    res[:, 0] = np.arccos((rng.random(vlen)-0.5)*2)
    res[:, 1] = rng.random(vlen)*2*np.pi
    return res


def check_pixangpix(vlen, ntry, nside, isnest):
    base = ph.Healpix_Base(nside, "NEST" if isnest else "RING")
    for cnt in range(ntry):
        inp = rng.integers(low=0, high=12*nside*nside-1, size=vlen)
        out = base.ang2pix(base.pix2ang(inp))
        if not np.array_equal(inp, out):
            raise ValueError("Test failed")


def check_vecpixvec(vlen, ntry, nside, isnest):
    base = ph.Healpix_Base(nside, "NEST" if isnest else "RING")
    for cnt in range(ntry):
        inp = ph.ang2vec(random_ptg(vlen))
        out = base.pix2vec(base.vec2pix(inp))
        if np.any(ph.v_angle(inp, out) > base.max_pixrad()):
            raise ValueError("Test failed")


def check_pixangvecpix(vlen, ntry, nside, isnest):
    base = ph.Healpix_Base(nside, "NEST" if isnest else "RING")
    for cnt in range(ntry):
        inp = rng.integers(low=0, high=12*nside*nside-1, size=vlen)
        out = base.vec2pix(ph.ang2vec(base.pix2ang(inp)))
        if not np.array_equal(inp, out):
            raise ValueError("Test failed")


def check_pixvecangpix(vlen, ntry, nside, isnest):
    base = ph.Healpix_Base(nside, "NEST" if isnest else "RING")
    for cnt in range(ntry):
        inp = rng.integers(low=0, high=12*nside*nside-1, size=vlen)
        out = base.ang2pix(ph.vec2ang(base.pix2vec(inp)))
        if not np.array_equal(inp, out):
            raise ValueError("Test failed")


def check_pixvecpix(vlen, ntry, nside, isnest):
    base = ph.Healpix_Base(nside, "NEST" if isnest else "RING")
    for cnt in range(ntry):
        inp = rng.integers(low=0, high=12*nside*nside-1, size=vlen)
        out = base.vec2pix(base.pix2vec(inp))
        if not np.array_equal(inp, out):
            raise ValueError("Test failed")


def check_ringnestring(vlen, ntry, nside):
    base = ph.Healpix_Base(nside, "NEST")
    for cnt in range(ntry):
        inp = rng.integers(low=0, high=12*nside*nside-1, size=vlen)
        out = base.nest2ring(base.ring2nest(inp))
        if not np.array_equal(inp, out):
            raise ValueError("Test failed")


def check_pixxyfpix(vlen, ntry, nside, isnest):
    base = ph.Healpix_Base(nside, "NEST" if isnest else "RING")
    for cnt in range(ntry):
        inp = rng.integers(low=0, high=12*nside*nside-1, size=vlen)
        out = base.xyf2pix(base.pix2xyf(inp))
        if not np.array_equal(inp, out):
            raise ValueError("Test failed")


def check_vecangvec(vlen, ntry):
    for cnt in range(ntry):
        inp = random_ptg(vlen)
        out = ph.vec2ang(ph.ang2vec(inp))
        if np.any(np.greater(np.abs(inp-out), 1e-10)):
            raise ValueError("Test failed")


check_vecangvec(1000, 1000)

for nside in (1, 32, 512, 8192, 32768*8):
    check_ringnestring(1000, 1000, nside)
    for isnest in (False, True):
        check_vecpixvec(1000, 1000, nside, isnest)
        check_pixangpix(1000, 1000, nside, isnest)
        check_pixvecpix(1000, 1000, nside, isnest)
        check_pixxyfpix(1000, 1000, nside, isnest)
        check_pixangvecpix(1000, 1000, nside, isnest)
        check_pixvecangpix(1000, 1000, nside, isnest)


isnest = False
for nside in (3, 7, 514, 8167, 32768*8+7):
    check_vecpixvec(1000, 1000, nside, isnest)
    check_pixangpix(1000, 1000, nside, isnest)
    check_pixvecpix(1000, 1000, nside, isnest)
    check_pixxyfpix(1000, 1000, nside, isnest)
    check_pixangvecpix(1000, 1000, nside, isnest)
    check_pixvecangpix(1000, 1000, nside, isnest)
