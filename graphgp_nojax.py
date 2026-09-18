import jax
jax.config.update("jax_enable_x64", True)
import graphgp
import numpy as np
import ducc0

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
n0=12
points = np.random.normal(size=(100000,3))

t0=time()
graph = graphgp.build_graph(points.copy(), n0=n0, k=k, cuda=False)
print("JAX time:", time()-t0)

import ducc0
t0=time()
ducc_points, ducc_nb, ducc_indices, ducc_offsets = ducc0.misc.experimental.build_graphgp(points.copy(),n0,k, nthreads=8)
print(np.max(np.abs(ducc_offsets-graph.offsets)))
print("ducc build graph:", time()-t0)
check_graph(ducc_points, ducc_nb, ducc_offsets)
print("max depth:",len(ducc_offsets)-1)

import matplotlib.pyplot as plt
plt.plot(ducc_offsets[1:]-ducc_offsets[:-1])
plt.show()
exit()
