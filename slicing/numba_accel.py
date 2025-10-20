"""Numba-accelerated kernels for performance-critical slicing routines.

Maintainer: Abdallah Kamhawi (PhD researcher, DART Laboratory; Kamhawi@umich.edu)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "HAS_NUMBA",
    "compute_face_grad_norms_jit",
    "extract_iso_segments_jit",
    "dijkstra_single_source_jit",
    "dijkstra_multi_source_jit",
    "dijkstra_with_cutoff_jit",
    "compute_adaptive_eps_jit",
    "classify_vertices_jit",
]

try:  # pragma: no cover - optional dependency
    from numba import njit, prange
except Exception:  # pragma: no cover - executed when numba is missing
    HAS_NUMBA = False

    def njit(*_args, **_kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator

    def prange(*args):  # type: ignore
        return range(*args)
else:  # pragma: no cover - executed only when numba is present
    HAS_NUMBA = True


if HAS_NUMBA:  # pragma: no cover - compiled at runtime

    @njit(cache=True, fastmath=True, parallel=True)
    def _compute_face_grad_norms_kernel(
        verts: np.ndarray,
        faces: np.ndarray,
        values: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        n_faces = faces.shape[0]
        out = np.zeros(n_faces, dtype=np.float64)

        for fi in prange(n_faces):
            i = faces[fi, 0]
            j = faces[fi, 1]
            k = faces[fi, 2]

            pix = verts[i, 0]
            piy = verts[i, 1]
            piz = verts[i, 2]

            pjx = verts[j, 0]
            pjy = verts[j, 1]
            pjz = verts[j, 2]

            pkx = verts[k, 0]
            pky = verts[k, 1]
            pkz = verts[k, 2]

            e1x = pjx - pix
            e1y = pjy - piy
            e1z = pjz - piz

            e2x = pkx - pix
            e2y = pky - piy
            e2z = pkz - piz

            n0 = e1y * e2z - e1z * e2y
            n1 = e1z * e2x - e1x * e2z
            n2 = e1x * e2y - e1y * e2x
            nlen = math.sqrt(n0 * n0 + n1 * n1 + n2 * n2)
            ulen = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
            if nlen < eps or ulen < eps:
                out[fi] = 0.0
                continue

            uhx = e1x / ulen
            uhy = e1y / ulen
            uhz = e1z / ulen

            dot_e2_u = e2x * uhx + e2y * uhy + e2z * uhz
            vtx = e2x - dot_e2_u * uhx
            vty = e2y - dot_e2_u * uhy
            vtz = e2z - dot_e2_u * uhz
            vlen = math.sqrt(vtx * vtx + vty * vty + vtz * vtz)
            if vlen < eps:
                out[fi] = 0.0
                continue

            vhx = vtx / vlen
            vhy = vty / vlen
            vhz = vtz / vlen

            xj = ulen
            xk = dot_e2_u
            yk = e2x * vhx + e2y * vhy + e2z * vhz

            if abs(xj) < eps or abs(yk) < eps:
                out[fi] = 0.0
                continue

            ri = values[i]
            rj = values[j]
            rk = values[k]

            a = (rj - ri) / xj
            b = (rk - ri - xk * a) / yk

            out[fi] = math.hypot(a, b)

        return out

    @njit(cache=True, fastmath=True, parallel=True)
    def _extract_iso_segments_kernel(
        verts: np.ndarray,
        faces: np.ndarray,
        values: np.ndarray,
        level: float,
        edge_tol: float,
        grad_norms: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_faces = faces.shape[0]
        seg_start = np.zeros((n_faces, 3), dtype=np.float64)
        seg_end = np.zeros((n_faces, 3), dtype=np.float64)
        mask = np.zeros(n_faces, dtype=np.uint8)
        lengths = np.zeros(n_faces, dtype=np.float64)
        weighted = np.zeros(n_faces, dtype=np.float64)

        for fi in prange(n_faces):
            idx0 = faces[fi, 0]
            idx1 = faces[fi, 1]
            idx2 = faces[fi, 2]

            pts = np.zeros((2, 3), dtype=np.float64)
            count = 0

            for edge in range(3):
                if edge == 0:
                    ia = idx0
                    ib = idx1
                elif edge == 1:
                    ia = idx1
                    ib = idx2
                else:
                    ia = idx2
                    ib = idx0

                fa = values[ia]
                fb = values[ib]
                da = fa - level
                db = fb - level
                if (da > edge_tol and db > edge_tol) or (da < -edge_tol and db < -edge_tol):
                    continue

                denom = fb - fa
                if math.fabs(denom) < edge_tol:
                    continue

                t = (level - fa) / denom
                if t < -edge_tol or t > 1.0 + edge_tol:
                    continue

                pax = verts[ia, 0]
                pay = verts[ia, 1]
                paz = verts[ia, 2]

                pbx = verts[ib, 0]
                pby = verts[ib, 1]
                pbz = verts[ib, 2]

                pts[count, 0] = pax + t * (pbx - pax)
                pts[count, 1] = pay + t * (pby - pay)
                pts[count, 2] = paz + t * (pbz - paz)
                count += 1

                if count == 2:
                    break

            if count == 2:
                sx = pts[0, 0]
                sy = pts[0, 1]
                sz = pts[0, 2]
                ex = pts[1, 0]
                ey = pts[1, 1]
                ez = pts[1, 2]

                dx = ex - sx
                dy = ey - sy
                dz = ez - sz
                seg_len = math.sqrt(dx * dx + dy * dy + dz * dz)

                seg_start[fi, 0] = sx
                seg_start[fi, 1] = sy
                seg_start[fi, 2] = sz
                seg_end[fi, 0] = ex
                seg_end[fi, 1] = ey
                seg_end[fi, 2] = ez

                lengths[fi] = seg_len
                g = grad_norms[fi]
                if math.isfinite(g):
                    weighted[fi] = g * seg_len
                mask[fi] = 1

        return seg_start, seg_end, mask, lengths, weighted

    @njit(cache=True)
    def _heap_push(heap_nodes, heap_dists, size, node, dist):
        i = size
        while i > 0:
            parent = (i - 1) // 2
            if heap_dists[parent] <= dist:
                break
            heap_nodes[i] = heap_nodes[parent]
            heap_dists[i] = heap_dists[parent]
            i = parent
        heap_nodes[i] = node
        heap_dists[i] = dist
        return size + 1

    @njit(cache=True)
    def _heap_pop(heap_nodes, heap_dists, size):
        node = heap_nodes[0]
        dist = heap_dists[0]
        size -= 1
        if size <= 0:
            return node, dist, size
        last_node = heap_nodes[size]
        last_dist = heap_dists[size]
        i = 0
        while True:
            left = 2 * i + 1
            if left >= size:
                break
            right = left + 1
            smallest = left
            if right < size and heap_dists[right] < heap_dists[left]:
                smallest = right
            if heap_dists[smallest] >= last_dist:
                break
            heap_nodes[i] = heap_nodes[smallest]
            heap_dists[i] = heap_dists[smallest]
            i = smallest
        heap_nodes[i] = last_node
        heap_dists[i] = last_dist
        return node, dist, size

    @njit(cache=True)
    def _dijkstra_kernel(
        indptr: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
        seeds: np.ndarray,
        seed_dists: np.ndarray,
        cutoff: float,
    ) -> np.ndarray:
        n = indptr.shape[0] - 1
        dist = np.empty(n, dtype=np.float64)
        visited = np.zeros(n, dtype=np.uint8)
        for i in range(n):
            dist[i] = np.inf

        heap_capacity = max(1, indices.shape[0] + seeds.shape[0] + 8)
        heap_nodes = np.empty(heap_capacity, dtype=np.int64)
        heap_dists = np.empty(heap_capacity, dtype=np.float64)
        heap_size = 0

        for i in range(seeds.shape[0]):
            node = seeds[i]
            seed_dist = seed_dists[i]
            if seed_dist < dist[node]:
                dist[node] = seed_dist
                heap_size = _heap_push(heap_nodes, heap_dists, heap_size, node, seed_dist)

        while heap_size > 0:
            node, cur_dist, heap_size = _heap_pop(heap_nodes, heap_dists, heap_size)
            if heap_size < 0:
                break
            if visited[node]:
                continue
            if cutoff >= 0.0 and cur_dist > cutoff:
                break
            visited[node] = 1
            if cur_dist > dist[node]:
                continue
            start = indptr[node]
            end = indptr[node + 1]
            for idx in range(start, end):
                nbr = indices[idx]
                weight = weights[idx]
                new_dist = cur_dist + weight
                if new_dist < dist[nbr]:
                    if cutoff >= 0.0 and new_dist > cutoff:
                        continue
                    dist[nbr] = new_dist
                    heap_size = _heap_push(heap_nodes, heap_dists, heap_size, nbr, new_dist)

        return dist

    @njit(cache=True)
    def _compute_adaptive_eps_kernel(
        field: np.ndarray,
        offsets: np.ndarray,
        ring_indices: np.ndarray,
        global_eps: float,
    ) -> np.ndarray:
        n = offsets.shape[0] - 1
        out = np.empty(n, dtype=np.float64)
        for v in range(n):
            start = offsets[v]
            end = offsets[v + 1]
            deg = end - start
            if deg < 2:
                out[v] = global_eps
                continue

            local_vals = np.empty(deg, dtype=np.float64)
            for i in range(deg):
                local_vals[i] = field[ring_indices[start + i]]

            local_min = local_vals[0]
            local_max = local_vals[0]
            for i in range(1, deg):
                val = local_vals[i]
                if val < local_min:
                    local_min = val
                if val > local_max:
                    local_max = val
            local_range = local_max - local_min

            pair_count = deg * (deg - 1) // 2
            if pair_count > 0:
                diffs = np.empty(pair_count, dtype=np.float64)
                idx = 0
                for i in range(deg):
                    vi = local_vals[i]
                    for j in range(i + 1, deg):
                        diffs[idx] = abs(vi - local_vals[j])
                        idx += 1
                diffs.sort()
                pos = int(0.1 * (pair_count - 1)) if pair_count > 1 else 0
                threshold = diffs[pos] if pair_count > 0 else 0.0
                candidate = threshold * 0.1
                other = local_range * 0.01
                adaptive = candidate if candidate < other else other
                if adaptive < global_eps:
                    adaptive = global_eps
                out[v] = adaptive
            else:
                adaptive = local_range * 0.01
                if adaptive < global_eps:
                    adaptive = global_eps
                out[v] = adaptive
        return out

    @njit(cache=True)
    def _classify_vertices_kernel(
        field: np.ndarray,
        offsets: np.ndarray,
        ring_indices: np.ndarray,
        boundary_mask: np.ndarray,
        eps_vec: np.ndarray,
        neighbor_counts: np.ndarray,
        allow_odd_as_saddle: bool,
        compute_confidence: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = offsets.shape[0] - 1
        class_codes = np.zeros(n, dtype=np.int8)
        confidences = np.ones(n, dtype=np.float64)

        for v in range(n):
            if boundary_mask[v] != 0:
                class_codes[v] = 0
                if compute_confidence:
                    confidences[v] = 0.0
                else:
                    confidences[v] = 1.0
                continue

            start = offsets[v]
            end = offsets[v + 1]
            length = end - start
            if length < 3:
                class_codes[v] = 0
                if compute_confidence:
                    confidences[v] = 0.0
                else:
                    confidences[v] = 1.0
                continue

            eps_v = eps_vec[v]
            diffs = np.empty(length, dtype=np.float64)

            first_sign = 0
            prev_sign = 0
            sign_changes = 0

            for i in range(length):
                nb = ring_indices[start + i]
                diff = field[v] - field[nb]
                diffs[i] = diff
                if abs(diff) > eps_v:
                    sign = 1 if diff > 0.0 else -1
                    if first_sign == 0:
                        first_sign = sign
                        prev_sign = sign
                    else:
                        if sign != prev_sign:
                            sign_changes += 1
                        prev_sign = sign

            if first_sign != 0 and prev_sign != first_sign:
                sign_changes += 1

            C = sign_changes
            classification = 0

            if C == 0:
                pos = False
                neg = False
                for i in range(length):
                    diff = diffs[i]
                    if diff > eps_v:
                        pos = True
                    elif diff < -eps_v:
                        neg = True
                if pos and not neg:
                    classification = 3
                elif neg and not pos:
                    classification = 1
            elif C == 2:
                classification = 0
            elif (C % 2 == 0 and C >= 4) or (allow_odd_as_saddle and C > 2):
                classification = 2
            else:
                classification = 0

            class_codes[v] = classification

            if compute_confidence:
                abs_diffs = np.empty(length, dtype=np.float64)
                significant_count = 0
                for i in range(length):
                    val = abs(diffs[i])
                    abs_diffs[i] = val
                    if val > eps_v:
                        significant_count += 1

                if significant_count == 0:
                    confidences[v] = 0.0
                    continue

                significant_vals = np.empty(significant_count, dtype=np.float64)
                idx_sig = 0
                for val in abs_diffs:
                    if val > eps_v:
                        significant_vals[idx_sig] = val
                        idx_sig += 1
                significant_vals.sort()
                mid = significant_count // 2
                if significant_count % 2 == 1:
                    signal_strength = significant_vals[mid]
                else:
                    signal_strength = 0.5 * (significant_vals[mid - 1] + significant_vals[mid])

                if signal_strength <= 0.0:
                    confidences[v] = 0.0
                    continue

                if C == 0:
                    sum_abs = 0.0
                    for val in abs_diffs:
                        sum_abs += val
                    mean_abs = sum_abs / length
                    mean_diff = 0.0
                    for val in diffs:
                        mean_diff += val
                    mean_diff /= length
                    var = 0.0
                    for val in diffs:
                        diff = val - mean_diff
                        var += diff * diff
                    var /= length
                    std_val = math.sqrt(var)
                    consistency = 1.0 - std_val / (mean_abs + 1e-10)
                    if consistency < 0.0:
                        consistency = 0.0
                elif C == 2:
                    pos = 0
                    neg = 0
                    for val in diffs:
                        if val > eps_v:
                            pos += 1
                        elif val < -eps_v:
                            neg += 1
                    if pos > 0 and neg > 0:
                        if pos > neg:
                            balance = neg / pos
                        else:
                            balance = pos / neg
                        consistency = balance
                    else:
                        consistency = 0.5
                elif C >= 4 and (C % 2) == 0:
                    consistency = 1.0 / (1.0 + abs(C - 4) * 0.1)
                else:
                    consistency = 0.0

                ring_quality = 1.0 if neighbor_counts[v] == length else 0.7

                confidence = signal_strength * consistency * ring_quality
                confidence = confidence / (signal_strength + 1.0)
                if confidence < 0.0:
                    confidence = 0.0
                if confidence > 1.0:
                    confidence = 1.0
                confidences[v] = confidence
            else:
                confidences[v] = 1.0

        return class_codes, confidences

else:  # pragma: no cover - executed without numba
    _compute_face_grad_norms_kernel = None
    _extract_iso_segments_kernel = None
    _dijkstra_kernel = None
    _compute_adaptive_eps_kernel = None
    _classify_vertices_kernel = None



def compute_face_grad_norms_jit(
    verts: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    eps: float,
) -> Optional[np.ndarray]:

    """Compute gradient magnitudes for every face in the mesh.
    
    Parameters
    ----------
    verts : np.ndarray
        Array of vertex positions shaped `(V, 3)`.
    faces : np.ndarray
        Triangle indices shaped `(F, 3)`.
    values : np.ndarray
        Scalar field sampled per vertex.
    eps : float
        Numerical guard used inside the kernel to avoid degenerate faces.
    
    Returns
    -------
    Optional[np.ndarray]
        `None` when numba is not available, otherwise an array of length `F`
        containing gradient magnitudes for each face."""
    if _compute_face_grad_norms_kernel is None:
        return None
    return _compute_face_grad_norms_kernel(verts, faces, values, float(eps))



def extract_iso_segments_jit(
    verts: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    level: float,
    edge_tol: float,
    grad_norms: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:

    """Extract iso-level intersection segments using the compiled kernel.
    
    Parameters
    ----------
    verts, faces, values : np.ndarray
        Mesh geometry and scalar field sampled per vertex.
    level : float
        Iso-value to intersect.
    edge_tol : float
        Tolerance applied when classifying edges as on the iso-level.
    grad_norms : np.ndarray
        Per-face gradient magnitudes used to filter weak intersections.
    
    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        Tuple of segment data as returned by the kernel, or `None` if numba
        is unavailable."""
    if _extract_iso_segments_kernel is None:
        return None
    return _extract_iso_segments_kernel(verts, faces, values, float(level), float(edge_tol), grad_norms)



def dijkstra_single_source_jit(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    source: int,
) -> Optional[np.ndarray]:

    """Compute single-source shortest paths using the compiled Dijkstra kernel.
    
    Parameters
    ----------
    indptr, indices, weights : np.ndarray
        CSR representation of the mesh graph.
    source : int
        Starting vertex index.
    
    Returns
    -------
    Optional[np.ndarray]
        Array of geodesic distances from `source` or `None` when numba is
        unavailable."""
    if _dijkstra_kernel is None:
        return None
    seeds = np.array([int(source)], dtype=np.int64)
    seed_dists = np.array([0.0], dtype=np.float64)
    return _dijkstra_kernel(indptr, indices, weights, seeds, seed_dists, -1.0)



def dijkstra_multi_source_jit(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    sources: np.ndarray,
) -> Optional[np.ndarray]:

    """Compute multi-source distances via the JIT-accelerated Dijkstra routine.
    
    Parameters
    ----------
    indptr, indices, weights : np.ndarray
        CSR representation of the mesh graph.
    sources : np.ndarray
        Array of seed vertices. An empty array yields `inf` everywhere.
    
    Returns
    -------
    Optional[np.ndarray]
        Distance array or `None` if the compiled kernel is missing."""
    if _dijkstra_kernel is None:
        return None
    if sources.size == 0:
        return np.full(indptr.shape[0] - 1, np.inf, dtype=np.float64)
    seeds = np.asarray(sources, dtype=np.int64)
    seed_dists = np.zeros(seeds.shape[0], dtype=np.float64)
    return _dijkstra_kernel(indptr, indices, weights, seeds, seed_dists, -1.0)



def dijkstra_with_cutoff_jit(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    source: int,
    cutoff: float,
) -> Optional[np.ndarray]:

    """Compute single-source distances with an optional cutoff radius.
    
    Parameters
    ----------
    indptr, indices, weights : np.ndarray
        CSR representation of the mesh graph.
    source : int
        Starting vertex index.
    cutoff : float
        Maximum distance to expand. `np.inf` yields the full solution.
    
    Returns
    -------
    Optional[np.ndarray]
        Distance array truncated at `cutoff` or `None` when numba is missing."""
    if _dijkstra_kernel is None:
        return None
    seeds = np.array([int(source)], dtype=np.int64)
    seed_dists = np.array([0.0], dtype=np.float64)
    return _dijkstra_kernel(indptr, indices, weights, seeds, seed_dists, float(cutoff))



def compute_adaptive_eps_jit(
    field: np.ndarray,
    offsets: np.ndarray,
    ring_indices: np.ndarray,
    global_eps: float,
) -> Optional[np.ndarray]:

    """Compute per-vertex adaptive epsilon values for critical-point detection.
    
    Parameters
    ----------
    field : np.ndarray
        Scalar field sampled per vertex.
    offsets : np.ndarray
        CSR row pointers describing one-ring neighbourhoods.
    ring_indices : np.ndarray
        CSR column indices for the one-ring neighbourhoods.
    global_eps : float
        Baseline epsilon value used when adaptation is disabled.
    
    Returns
    -------
    Optional[np.ndarray]
        Array of per-vertex epsilon values or `None` if numba is missing."""
    if _compute_adaptive_eps_kernel is None:
        return None
    return _compute_adaptive_eps_kernel(field, offsets, ring_indices, float(global_eps))



def classify_vertices_jit(
    field: np.ndarray,
    offsets: np.ndarray,
    ring_indices: np.ndarray,
    boundary_mask: np.ndarray,
    eps_vec: np.ndarray,
    neighbor_counts: np.ndarray,
    allow_odd_as_saddle: bool,
    compute_confidence: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

    """Classify vertices via the compiled saddle/critical-point detector.
    
    Parameters
    ----------
    field : np.ndarray
        Scalar field sampled per vertex.
    offsets, ring_indices : np.ndarray
        CSR representation of the one-ring connectivity.
    boundary_mask : np.ndarray
        Boolean array marking vertices on the boundary.
    eps_vec : np.ndarray
        Per-vertex epsilon thresholds (often from :func:compute_adaptive_eps_jit).
    neighbor_counts : np.ndarray
        Number of neighbours for each vertex.
    allow_odd_as_saddle : bool
        Whether odd sign-change counts above two should be classified as saddles.
    compute_confidence : bool
        If `True`, the kernel also returns confidence scores per vertex.
    
    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        Tuple of `(class_codes, confidences)` where each element is `None`
        when the kernel is unavailable."""
    if _classify_vertices_kernel is None:
        return None, None
    return _classify_vertices_kernel(
        field,
        offsets,
        ring_indices,
        boundary_mask,
        eps_vec,
        neighbor_counts,
        allow_odd_as_saddle,
        compute_confidence,
    )

