import numpy as np
import ducc_0_1.fft as fft


def _l2error(a, b, axes):
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(a)**2))/np.log2(np.max([2,np.prod(np.take(a.shape,axes))]))


def fftn(a, axes=None, inorm=0, out=None, nthreads=1):
    return fft.c2c(a, axes=axes, forward=True, inorm=inorm,
                   out=out, nthreads=nthreads)


def ifftn(a, axes=None, inorm=0, out=None, nthreads=1):
    return fft.c2c(a, axes=axes, forward=False, inorm=inorm,
                   out=out, nthreads=nthreads)


def rfftn(a, axes=None, inorm=0, nthreads=1):
    return fft.r2c(a, axes=axes, forward=True, inorm=inorm,
                   nthreads=nthreads)


def irfftn(a, axes=None, lastsize=0, inorm=0, nthreads=1):
    return fft.c2r(a, axes=axes, lastsize=lastsize, forward=False,
                   inorm=inorm, nthreads=nthreads)


nthreads = 0


def update_err(err, name, value, shape):
    if name in err and err[name] >= value:
        return err
    err[name] = value
    print(shape)
    for (nm, v) in err.items():
        print("{}: {}".format(nm, v))
    print()
    return err


def test(err):
    ndim = np.random.randint(1, 5)
    axlen = int((2**20)**(1./ndim))
    shape = np.random.randint(1, axlen, ndim)
    axes = np.arange(ndim)
    np.random.shuffle(axes)
    nax = np.random.randint(1, ndim+1)
    axes = axes[:nax]
    lastsize = shape[axes[-1]]
    a = np.random.rand(*shape)-0.5 + 1j*np.random.rand(*shape)-0.5j
    a_32 = a.astype(np.complex64)
    b = ifftn(fftn(a, axes=axes, nthreads=nthreads), axes=axes, inorm=2,
              nthreads=nthreads)
    err = update_err(err, "cmax", _l2error(a, b, axes), shape)
    b = ifftn(fftn(a.real, axes=axes, nthreads=nthreads), axes=axes, inorm=2,
              nthreads=nthreads)
    err = update_err(err, "cmax", _l2error(a.real, b, axes), shape)
    b = fftn(ifftn(a.real, axes=axes, nthreads=nthreads), axes=axes, inorm=2,
             nthreads=nthreads)
    err = update_err(err, "cmax", _l2error(a.real, b, axes), shape)
    b = ifftn(fftn(a.astype(np.complex64), axes=axes, nthreads=nthreads),
              axes=axes, inorm=2, nthreads=nthreads)
    err = update_err(err, "cmaxf", _l2error(a.astype(np.complex64), b, axes), shape)
    b = irfftn(rfftn(a.real, axes=axes, nthreads=nthreads), axes=axes, inorm=2,
               lastsize=lastsize, nthreads=nthreads)
    err = update_err(err, "rmax", _l2error(a.real, b, axes), shape)
    b = irfftn(rfftn(a.real.astype(np.float32), axes=axes, nthreads=nthreads),
               axes=axes, inorm=2, lastsize=lastsize, nthreads=nthreads)
    err = update_err(err, "rmaxf", _l2error(a.real.astype(np.float32), b, axes), shape)
    b = fft.separable_hartley(
        fft.separable_hartley(a.real, axes=axes, nthreads=nthreads),
        axes=axes, inorm=2, nthreads=nthreads)
    err = update_err(err, "hmax", _l2error(a.real, b, axes), shape)
    b = fft.genuine_hartley(
        fft.genuine_hartley(a.real, axes=axes, nthreads=nthreads),
        axes=axes, inorm=2, nthreads=nthreads)
    err = update_err(err, "hmax", _l2error(a.real, b, axes), shape)
    b = fft.separable_hartley(
            fft.separable_hartley(
                a.real.astype(np.float32), axes=axes, nthreads=nthreads),
            axes=axes, inorm=2, nthreads=nthreads)
    err = update_err(err, "hmaxf", _l2error(a.real.astype(np.float32), b, axes), shape)
    b = fft.genuine_hartley(
            fft.genuine_hartley(a.real.astype(np.float32), axes=axes,
                                        nthreads=nthreads),
            axes=axes, inorm=2, nthreads=nthreads)
    err = update_err(err, "hmaxf", _l2error(a.real.astype(np.float32), b, axes), shape)
    if all(a.shape[i] > 1 for i in axes):
        b = fft.dct(
            fft.dct(a.real, axes=axes, nthreads=nthreads, type=1),
            axes=axes, type=1, nthreads=nthreads, inorm=2)
        err = update_err(err, "c1max", _l2error(a.real, b, axes), shape)
        b = fft.dct(
            fft.dct(a_32.real, axes=axes, nthreads=nthreads, type=1),
            axes=axes, type=1, nthreads=nthreads, inorm=2)
        err = update_err(err, "c1maxf", _l2error(a_32.real, b, axes), shape)
    b = fft.dct(
        fft.dct(a.real, axes=axes, nthreads=nthreads, type=2),
        axes=axes, type=3, nthreads=nthreads, inorm=2)
    err = update_err(err, "c23max", _l2error(a.real, b, axes), shape)
    b = fft.dct(
        fft.dct(a_32.real, axes=axes, nthreads=nthreads, type=2),
        axes=axes, type=3, nthreads=nthreads, inorm=2)
    err = update_err(err, "c23maxf", _l2error(a_32.real, b, axes), shape)
    b = fft.dct(
        fft.dct(a.real, axes=axes, nthreads=nthreads, type=4),
        axes=axes, type=4, nthreads=nthreads, inorm=2)
    err = update_err(err, "c4max", _l2error(a.real, b, axes), shape)
    b = fft.dct(
        fft.dct(a_32.real, axes=axes, nthreads=nthreads, type=4),
        axes=axes, type=4, nthreads=nthreads, inorm=2)
    err = update_err(err, "c4maxf", _l2error(a_32.real, b, axes), shape)
    b = fft.dst(
        fft.dst(a.real, axes=axes, nthreads=nthreads, type=1),
        axes=axes, type=1, nthreads=nthreads, inorm=2)
    err = update_err(err, "s1max", _l2error(a.real, b, axes), shape)
    b = fft.dst(
        fft.dst(a_32.real, axes=axes, nthreads=nthreads, type=1),
        axes=axes, type=1, nthreads=nthreads, inorm=2)
    err = update_err(err, "s1maxf", _l2error(a_32.real, b, axes), shape)
    b = fft.dst(
        fft.dst(a.real, axes=axes, nthreads=nthreads, type=2),
        axes=axes, type=3, nthreads=nthreads, inorm=2)
    err = update_err(err, "s23max", _l2error(a.real, b, axes), shape)
    b = fft.dst(
        fft.dst(a_32.real, axes=axes, nthreads=nthreads, type=2),
        axes=axes, type=3, nthreads=nthreads, inorm=2)
    err = update_err(err, "s23maxf", _l2error(a_32.real, b, axes), shape)
    b = fft.dst(
        fft.dst(a.real, axes=axes, nthreads=nthreads, type=4),
        axes=axes, type=4, nthreads=nthreads, inorm=2)
    err = update_err(err, "s4max", _l2error(a.real, b, axes), shape)
    b = fft.dst(
        fft.dst(a_32.real, axes=axes, nthreads=nthreads, type=4),
        axes=axes, type=4, nthreads=nthreads, inorm=2)
    err = update_err(err, "s4maxf", _l2error(a_32.real, b, axes), shape)


err = dict()
while True:
    test(err)
