import numpy as np

def build_tree(points):
    def build_tree_sub(lo, hi):
        if hi <= lo+1:  # interval length <=1, done
            return
        ploc = points[lo:hi]
#        ext = np.minmax(ploc, axis=0)
#        splitdim = np.argmax(ext[1]-ext[0])
        splitdim = np.argmax(np.max(ploc, axis=0)-np.min(ploc, axis=0))
        kth = (hi-lo)//2
        idx = np.argpartition(ploc[:,splitdim], kth)
        ploc[()] = ploc[idx]  # reorder
        indices[lo:hi] = indices[lo:hi][idx]
        build_tree_sub(lo, lo+kth)
        build_tree_sub(lo+kth+1, hi)
        split_dims[lo+kth] = splitdim

    points_new = points.copy()
    indices = np.arange(len(points), dtype=np.int32)
    split_dims = np.full(len(points), 255, dtype=np.uint8)
    build_tree_sub(0, len(points))
    return points, split_dims, indices

def rearrange(points, split_dims, indices):
    idx = np.zeros(len(points), dtype=np.int32)

    def recurse(lo, hi, pos, lvl):
        if lo >= hi:
            return
        kth = (hi-lo)//2
        idx[pos] = lo + kth
        recurse(lo, lo+kth, pos + (1<<lvl), lvl+1)
        recurse(lo+kth+1, hi, pos + (2<<lvl), lvl+1)

    recurse(0, len(points), 0, 0)
    return points[idx], split_dims[idx], indices[idx]

def find_neighbors(points, split_dims, n0, k):
    npnt = len(points)
    nb = np.full((npnt-n0, k), -1, dtype=np.int32)
    rsqbuf = np.full((k,), 1e300, dtype=np.float64)
    idxbuf = np.full((k,), -1, dtype=np.int32)

    def step(i, pos, rsqbuf, idxbuf, lvl):
       # investigate the current point
       if pos < i:  # we are allowed to use this point
           loc = points[i]
           vdist = loc-points[pos]
           dsq = np.vdot(vdist,vdist)
           if dsq < rsqbuf[-1]:
               rsqbuf[-1] = dsq
               idxbuf[-1] = pos
               tmpidx = np.argsort(rsqbuf)
               rsqbuf[()] = rsqbuf[tmpidx]
               idxbuf[()] = idxbuf[tmpidx]
           # on which side of the dividing plane are we?
           spl = split_dims[pos]
           if spl == 255:
               return
           planedist = loc[spl] - points[pos,spl]
           pos_left = pos + (1<<lvl)
           pos_right = pos + (2<<lvl)
           if planedist <= 0:  # we are in the lower part
               step(i, pos_left, rsqbuf, idxbuf, lvl+1)
               if rsqbuf[-1] > planedist**2:
                   step(i, pos_right, rsqbuf, idxbuf, lvl+1)
           else:
               step(i, pos_right, rsqbuf, idxbuf, lvl+1)
               if rsqbuf[-1] > planedist**2:
                   step(i, pos_left, rsqbuf, idxbuf, lvl+1)

    if False:

        def recurse(lo, hi, pos, lvl):
            # print("pos", pos)
            if lo >= hi:
                return
            kth = (hi-lo)//2
            # deal with the midpoint kth, pos
            if pos >= n0:  # we need to search neighbors
                rsqbuf[()] = 1e300
                idxbuf[()] = -1
                step(pos, 0, rsqbuf, idxbuf, 0)
                nb[pos-n0] = idxbuf
            recurse(lo, lo+kth, pos + (1<<lvl), lvl+1)
            recurse(lo+kth+1, hi, pos + (2<<lvl), lvl+1)

        recurse(0, len(points), 0, 0)

    else:
        for i in range(n0, npnt):
            rsqbuf[()] = 1e300
            idxbuf[()] = -1
            step(i, 0, rsqbuf, idxbuf, 0)
            nb[i-n0] = idxbuf
    return nb

def compute_depths(nb, n0):
    depths = np.zeros(n0+len(nb), dtype=np.int32)+1000000000
    depths[:n0] = 0
    for i in range(n0, len(depths)):
        depths[i] = 1 + np.max(depths[nb[i-n0]])
    return depths

def order_by_depth(points, indices, neighbors, depths):
    n0 = len(points) - len(neighbors)
    order = np.argsort(depths)
    points, indices, depths = points[order], indices[order], depths[order]
    neighbors = neighbors[order[n0:] - n0]  # first n0 should stay in order
    inv_order = np.arange(len(points), dtype=int)
    inv_order[order] = inv_order
    neighbors = inv_order[neighbors]
    return points, indices, neighbors, depths


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

from time import time
np.random.seed(42)
k=4
n0=50
points = np.random.normal(size=(10000000,3))
# opoints = points.copy()
# t0=time()
# points, split_dims, indices = build_tree(points)
# print("build tree:", time()-t0)
# t0=time()
# points, split_dims, indices = rearrange(points, split_dims, indices)
# print("rearrange:", time()-t0)
# t0=time()
# nb = find_neighbors(points, split_dims, n0, k)
# print("neighbors:", time()-t0)
# t0=time()
# depths = compute_depths(nb, n0)
# print("depths:", time()-t0)
# t0=time()
# points, indices, nb, depths = order_by_depth(points, indices, nb, depths)
# print("order by depth:", time()-t0)
# t0=time()
# print("max depth:",np.max(depths))
# offsets = np.searchsorted(depths, np.arange(1, np.max(depths) + 2))
# check_graph(points, nb, offsets)

import ducc0
t0=time()
ducc_points, ducc_nb, ducc_indices, ducc_depths = ducc0.misc.experimental.build_graphgp(points,n0,k, nthreads=1)
ducc_points, ducc_indices, ducc_nb, ducc_depths = order_by_depth(ducc_points, ducc_indices, ducc_nb, ducc_depths)
ducc_offsets = np.searchsorted(ducc_depths, np.arange(1, np.max(ducc_depths) + 2))
print("ducc build graph:", time()-t0)
check_graph(ducc_points, ducc_nb, ducc_offsets)
print("max depth:",np.max(ducc_depths))
print(np.max(np.abs(depths-ducc_depths)))
print(np.max(np.abs(points-ducc_points)))

import matplotlib.pyplot as plt
counts, bins = np.histogram(depths)
plt.stairs(counts, bins)
counts, bins = np.histogram(ducc_depths)
plt.stairs(counts*1.1, bins)
plt.show()
exit()
