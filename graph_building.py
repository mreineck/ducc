import nifty.re as jft
import graphgp as gp
from jax import numpy as jnp
import numpy as np
import healpy as hp
import ducc0

# create Healpix-LogR Grid

# change nside and/or rminshape for smaller grid and less memory usage
nside = 256
rminshape = 512

# other parameters can stay fixed
nside0 = 1
rmin = 1e-1
rmax = 1250
rlinthresh = 500
grid = jft.HPBrokenLogRGrid(nside=nside, nside0=nside0, r_min_shape=rminshape, r_min=rmin,
                            r_max=rmax, r_linthresh=rlinthresh)

# Calculate points from grid
grid_logr = grid.grids[1].at(-1)
nrad = grid_logr.shape[0]
idxs = jnp.arange(nrad)[jnp.newaxis, :]
coords = grid_logr.index2coord(idxs)  # center of pixel
r_center = coords[0]
vec = np.asarray(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)), nest=True))  # (3, npix)
npix, nr = vec.shape[1], len(r_center)
points = np.empty((npix, nr, 3), dtype=np.float64)
for j in range(3):
    np.multiply(vec[j][:, None], r_center[None, :], out=points[:, :, j])
points = points.reshape(-1, 3)
print(points.dtype)
points=points.astype(np.float32)

def check_graph(points, neighbors, offsets):
    # Ensure offsets are valid
    assert offsets[0] == len(points) - len(neighbors), "Neighbors should start from first offset"
    assert np.all(offsets[1:] >= offsets[:-1]), "Offsets must be non-decreasing"
    assert offsets[-1] <= len(points), "Last offset must be less than or equal to the number of points"

    # Ensure topological order
    max_neighbors = np.max(neighbors, axis=1)
    index = np.arange(len(neighbors)) + offsets[0]
    ok = max_neighbors < index
    assert np.all(ok), "Points are not in topological order"

    # Ensure only coarse points
    offsets_index = np.searchsorted(offsets, index, side="right") - 1
    assert np.all(max_neighbors < offsets[offsets_index]), "Neighbors must not be in the same batch"

# build graph
#graph = gp.build_graph(points, n0=100, k=12, cuda=False)
print("# of points:",len(points))
from time import time
t0=time()
ducc_points, ducc_nb, ducc_indices, ducc_offsets = ducc0.misc.experimental.build_graphgp(points,n0=100,k=12, nthreads=8)
print("ducc build graph:", time()-t0)
check_graph(ducc_points, ducc_nb, ducc_offsets)
print("max depth:",len(ducc_offsets)-1)
