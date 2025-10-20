"""Utility primitives for querying geometric scales on surface meshes.

Maintainer: Abdallah Kamhawi (PhD researcher, DART Laboratory; Kamhawi@umich.edu)
"""

from __future__ import annotations

import numpy as np

from loaders.mesh_loader import MeshLoader




def _median_edge_length(mesh: MeshLoader) -> float:



    """Estimate a characteristic edge length to parameterise geodesic filters.

    The routine prefers the median of the unique edge lengths because the
    median is robust to a small number of outlier edges. When the mesh no
    longer exposes unique edges (for example after boolean operations) the
    function falls back to a nearest-neighbour heuristic measured on the
    vertex set.

    Parameters
    ----------
    mesh : MeshLoader
        Surface mesh already loaded into memory.

    Returns
    -------
    float
        Characteristic distance scale measured in mesh units.
    """
    try:
        edges = mesh.edges_unique
        if edges is None or len(edges) == 0:
            raise ValueError

        v0 = mesh.vertices[edges[:, 0]]
        v1 = mesh.vertices[edges[:, 1]]
        lengths = np.linalg.norm(v1 - v0, axis=1)
        lengths = lengths[np.isfinite(lengths)]
        if lengths.size == 0:
            raise ValueError
        return float(np.median(lengths))
    except Exception:
        # Fallback: median nearest-neighbour spacing across a capped vertex subset.
        verts = mesh.vertices
        if verts.shape[0] < 2:
            return 1.0

        sample_idx = np.arange(min(verts.shape[0], 5000))
        cloud = verts[sample_idx]
        dmin = np.full(cloud.shape[0], np.inf, dtype=float)

        # Simple O(n^2) loop (bounded by 5k samples) keeps the implementation dependency free.
        for i in range(cloud.shape[0]):
            distances = np.linalg.norm(cloud - cloud[i], axis=1)
            distances[i] = np.inf
            dmin[i] = float(np.min(distances))

        filtered = dmin[np.isfinite(dmin)]
        return float(np.median(filtered)) if filtered.size else 1.0


