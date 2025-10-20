"""Critical-point detection utilities for scalar fields on triangle meshes.

The implementation follows the piecewise-linear upper/lower link test by
counting sign changes around each vertex. Optional Laplacian smoothing, adaptive
epsilon thresholds, and persistence-based filtering make the classifier
robust to noisy meshes while keeping the underlying mathematics explicit.

Maintainer: Abdallah Kamhawi (PhD researcher, DART Laboratory; Kamhawi@umich.edu)
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Sequence
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from slicing.numba_accel import (
    HAS_NUMBA,
    classify_vertices_jit,
    compute_adaptive_eps_jit,
    dijkstra_with_cutoff_jit,
)

# ----------------------------- topology -------------------------------- #
def _build_vertex_ring_and_links(
    faces: np.ndarray,
    V: int,
    *,
    show_progress: bool = True
) -> Tuple[List[Set[int]], List[Set[Tuple[int,int]]]]:
    """Construct first-ring neighbourhoods and edge links for every vertex.
    
    Parameters
    ----------
    faces : np.ndarray
        Triangle indices shaped `(F, 3)`.
    V : int
        Number of vertices in the mesh.
    show_progress : bool, optional
        Display a progress bar while scanning the faces.
    
    Returns
    -------
    Tuple[List[Set[int]], List[Set[Tuple[int, int]]]]
        Adjacency structures describing one-ring neighbours and link edges."""
    neighbors: List[Set[int]] = [set() for _ in range(V)]
    link_pairs: List[Set[Tuple[int,int]]] = [set() for _ in range(V)]
    F = np.asarray(faces, dtype=int)
    for tri in tqdm(
        F,
        total=F.shape[0],
        desc="Building 1-ring structures",
        unit="face",
        disable=not show_progress,
    ):
        i, j, k = tri
        neighbors[i].update((j,k)); neighbors[j].update((i,k)); neighbors[k].update((i,j))
        link_pairs[i].add((min(j,k), max(j,k)))
        link_pairs[j].add((min(i,k), max(i,k)))
        link_pairs[k].add((min(i,j), max(i,j)))
    return neighbors, link_pairs

def _infer_boundary_mask_from_faces(
    faces: np.ndarray,
    V: int,
    *,
    show_progress: bool = True
) -> np.ndarray:
    """Infer boundary vertices by counting face incidences per edge.
    
    Parameters
    ----------
    faces : np.ndarray
        Triangle indices shaped `(F, 3)`.
    V : int
        Number of vertices in the mesh.
    show_progress : bool, optional
        Display a progress bar while traversing faces.
    
    Returns
    -------
    np.ndarray
        Boolean mask where `True` marks boundary vertices."""
    F = np.asarray(faces, dtype=int)
    ec = defaultdict(int)
    for tri in tqdm(
        F,
        total=F.shape[0],
        desc="Inferring boundary vertices",
        unit="face",
        disable=not show_progress,
    ):
        i, j, k = tri
        for a,b in ((i,j),(j,k),(k,i)):
            if a>b: a,b=b,a
            ec[(a,b)] += 1
    mask = np.zeros(V, dtype=bool)
    for (a,b), c in ec.items():
        if c == 1:
            mask[a] = True; mask[b] = True
    return mask

# -------------------------- geometric ordering ------------------------- #
def _vertex_normal_from_ring(v: int, faces: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Compute an area-weighted normal around vertex `v` with a PCA fallback.
    
    Parameters
    ----------
    v : int
        Vertex index whose normal is requested.
    faces : np.ndarray
        Triangle indices shaped `(F, 3)`.
    verts : np.ndarray
        Vertex coordinates shaped `(V, 3)`.
    
    Returns
    -------
    np.ndarray
        Unit normal vector approximated from the incident faces or PCA fallback."""
    idx = np.where((faces == v).any(axis=1))[0]
    if idx.size == 0:
        return np.array([0.0,0.0,1.0])
    nsum = np.zeros(3, dtype=float)
    for fi in idx:
        i,j,k = faces[fi]
        p0,p1,p2 = verts[i], verts[j], verts[k]
        nsum += np.cross(p1-p0, p2-p0)
    nlen = np.linalg.norm(nsum)
    if nlen < 1e-14:
        ring = np.unique(faces[idx].ravel())
        offs = verts[ring] - verts[v]
        C = offs.T @ offs
        w,U = np.linalg.eigh(C)
        n = U[:,0]
        return n / (np.linalg.norm(n)+1e-15)
    return nsum / nlen

def _tangent_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal tangent basis for the supplied normal.
    
    Parameters
    ----------
    n : np.ndarray
        Surface normal vector.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Pair of unit vectors spanning the tangent plane."""
    n = n / (np.linalg.norm(n)+1e-15)
    a = np.array([1.0,0.0,0.0])
    if abs(float(np.dot(a,n))) > 0.9: a = np.array([0.0,1.0,0.0])
    e1 = a - float(np.dot(a,n))*n; e1 /= (np.linalg.norm(e1)+1e-15)
    e2 = np.cross(n,e1); e2 /= (np.linalg.norm(e2)+1e-15)
    return e1, e2

def _order_neighbors_by_angle(v: int,
                              nbs: List[int],
                              faces: np.ndarray,
                              verts: np.ndarray) -> List[int]:
    """Order one-ring neighbours of `v` by polar angle in the tangent plane.
    
    Parameters
    ----------
    v : int
        Central vertex index.
    ring : Sequence[int]
        Neighbour vertex indices.
    faces : np.ndarray
        Triangle indices.
    verts : np.ndarray
        Vertex coordinates.
    
    Returns
    -------
    List[int]
        Neighbour list sorted counter-clockwise in the tangent plane."""
    if len(nbs) <= 2:
        return list(nbs)
    pv = verts[v]
    n = _vertex_normal_from_ring(v, faces, verts)
    e1, e2 = _tangent_basis(n)
    offs = verts[nbs] - pv
    x = offs @ e1; y = offs @ e2
    ang = np.arctan2(y, x)
    order = np.argsort(ang, kind='mergesort')  # stable
    return [int(nbs[i]) for i in order]

def _ordered_one_ring(v: int,
                      neighbors: List[Set[int]],
                      link_pairs: List[Set[Tuple[int,int]]],
                      faces: Optional[np.ndarray],
                      verts: Optional[np.ndarray]) -> List[int]:
    """Return the one-ring around `v` ordered by angle.
    
    Parameters
    ----------
    v : int
        Central vertex index.
    ring_lists : Sequence[Set[int]]
        Raw one-ring adjacency.
    faces : np.ndarray
        Triangle indices.
    verts : np.ndarray
        Vertex coordinates.
    
    Returns
    -------
    List[int]
        Angle-ordered neighbours of `v`."""
    nset = neighbors[v]
    if not nset:
        return []
    nbs = list(nset)

    # Build adjacency among neighbors using link edges
    adj = {u:set() for u in nbs}
    for a,b in link_pairs[v]:
        if a in adj and b in adj:
            adj[a].add(b); adj[b].add(a)

    # Try to traverse a cycle/path covering all neighbors exactly once
    start = nbs[0]
    deg1 = [u for u in nbs if len(adj[u]) == 1]
    if deg1:
        start = deg1[0]
    order = [start]
    prev = None
    cur = start
    seen = {start}
    while True:
        nxts = [w for w in adj[cur] if w != prev]
        nxt = None
        for w in nxts:
            if w not in seen:
                nxt = w; break
        if nxt is None:
            if len(seen) == len(nbs):
                break
            else:
                order = None
                break
        order.append(nxt); seen.add(nxt); prev,cur = cur,nxt
        if len(order) > len(nbs) + 2:
            order = None
            break

    if order is None or len(order) != len(nbs):
        if verts is not None and faces is not None:
            return _order_neighbors_by_angle(v, nbs, faces, verts)
        else:
            return sorted(nbs)
    return order

# -------------------------- smoothing ---------------------------------- #
def _laplacian_smooth_field(
    field: np.ndarray,
    neighbors: List[Set[int]],
    boundary_mask: np.ndarray,
    iterations: int = 2,
    weight: float = 0.5,
    *,
    show_progress: bool = True
) -> np.ndarray:
    """Apply explicit Laplacian smoothing to a scalar field.
    
    Parameters
    ----------
    field : np.ndarray
        Scalar field sampled per vertex.
    neighbors : List[Set[int]]
        One-ring adjacency lists.
    boundary_mask : np.ndarray
        Boolean array marking boundary vertices.
    iterations : int
        Number of smoothing passes.
    weight : float
        Relaxation weight applied each iteration.
    show_progress : bool, optional
        Display a progress bar during smoothing.
    
    Returns
    -------
    np.ndarray
        Smoothed scalar field."""
    smooth = field.copy()
    for it in tqdm(
        range(iterations),
        desc="Smoothing field (Laplacian)",
        unit="iter",
        disable=not show_progress,
    ):
        new_vals = smooth.copy()
        for v in tqdm(
            range(len(field)),
            desc=f"  iter {it+1}/{iterations}",
            unit="vtx",
            leave=False,
            disable=not show_progress,
        ):
            if boundary_mask[v]:
                continue  # Keep boundary values fixed
            nbs = neighbors[v]
            if nbs:
                avg = np.mean([smooth[n] for n in nbs])
                new_vals[v] = (1 - weight) * smooth[v] + weight * avg
        smooth = new_vals
    return smooth

# -------------------------- adaptive epsilon --------------------------- #
def _compute_adaptive_eps(
    v: int,
    field: np.ndarray,
    neighbors: List[Set[int]],
    global_eps: float = 1e-12,
    percentile: float = 10.0
) -> float:
    """Compute a per-vertex epsilon value using local variation statistics.
    
    Parameters
    ----------
    v : int
        Vertex index being analysed.
    field : np.ndarray
        Scalar field sampled per vertex.
    neighbors : List[Set[int]]
        One-ring adjacency.
    global_eps : float
        Baseline epsilon.
    percentile : float
        Percentile of absolute differences used to scale the epsilon.
    
    Returns
    -------
    float
        Adaptive epsilon assigned to vertex `v`."""
    nbs = list(neighbors[v])
    if len(nbs) < 2:
        return global_eps
    
    neighbor_vals = field[nbs]
    local_range = float(np.ptp(neighbor_vals))
    
    diffs = []
    for i in range(len(nbs)):
        for j in range(i+1, len(nbs)):
            diffs.append(abs(float(field[nbs[i]] - field[nbs[j]])))
    
    if diffs:
        threshold = np.percentile(diffs, percentile)
        adaptive = max(global_eps, min(threshold * 0.1, local_range * 0.01))
    else:
        adaptive = max(global_eps, local_range * 0.01)
    
    return float(adaptive)

# --------------------------- sign changes ------------------------------ #
def _count_sign_changes_closed(vals: List[float], eps: float) -> int:
    """Count sign changes in a cyclic sequence with tolerance `eps`.
    
    Parameters
    ----------
    vals : List[float]
        Differences `f(v) - f(u_i)` around the one-ring.
    eps : float
        Magnitude below which values are treated as zero.
    
    Returns
    -------
    int
        Twice the number of alternating up/down sectors."""
    s = [x for x in vals if abs(x) > eps]
    if len(s) == 0:
        return 0
    s.append(s[0])
    cnt = 0
    prev = s[0]
    for x in s[1:]:
        if prev * x < 0.0:
            cnt += 1
        prev = x
    return cnt

# -------------------------- confidence scoring ------------------------- #
def _compute_confidence_score(
    v: int,
    field: np.ndarray,
    neighbors: List[Set[int]],
    ring: List[int],
    C: int,
    eps: float
) -> float:
    """Estimate a confidence score for the classification at vertex `v`.
    
    Parameters
    ----------
    v : int
        Central vertex index.
    field : np.ndarray
        Scalar field sampled per vertex.
    neighbors : List[Set[int]]
        One-ring adjacency.
    ring : List[int]
        Angle-ordered one-ring for `v`.
    C : int
        Sign-change count.
    eps : float
        Local tolerance.
    
    Returns
    -------
    float
        Confidence value in `[0, 1]`."""
    if len(ring) < 3:
        return 0.0
    
    fv = float(field[v])
    diffs = np.array([float(field[u] - fv) for u in ring])
    
    abs_diffs = np.abs(diffs)
    significant = abs_diffs[abs_diffs > eps]
    if len(significant) > 0:
        signal_strength = float(np.median(significant))
    else:
        return 0.0
    
    if C == 0:
        consistency = 1.0 - float(np.std(diffs) / (np.mean(abs_diffs) + 1e-10))
    elif C == 2:
        pos = diffs > eps
        neg = diffs < -eps
        if np.sum(pos) > 0 and np.sum(neg) > 0:
            balance = min(np.sum(pos), np.sum(neg)) / max(np.sum(pos), np.sum(neg))
            consistency = float(balance)
        else:
            consistency = 0.5
    elif C >= 4 and C % 2 == 0:
        expected_sector_size = len(ring) / C
        consistency = 1.0 / (1.0 + abs(C - 4) * 0.1)
    else:
        consistency = 0.0
    
    ring_quality = 1.0 if len(ring) == len(neighbors[v]) else 0.7
    
    confidence = signal_strength * consistency * ring_quality
    confidence = float(np.clip(confidence / (signal_strength + 1.0), 0.0, 1.0))
    
    return confidence

# -------------------------- persistence --------------------------------- #
def _compute_persistence(
    field: np.ndarray,
    critical_points: Dict[str, List[Tuple[int, float]]],
    neighbors: List[Set[int]],
    *,
    show_progress: bool = True
) -> Dict[int, float]:
    """Approximate topological persistence for critical points using geodesic distances.
    
    Parameters
    ----------
    field : np.ndarray
        Scalar field sampled per vertex.
    critical_points : Dict[str, List[Tuple[int, float]]]
        Classified critical points by type.
    neighbors : List[Set[int]]
        One-ring adjacency.
    show_progress : bool, optional
        Show progress while computing persistence disks.
    
    Returns
    -------
    Dict[int, float]
        Persistence estimate per critical vertex."""
    persistence = {}
    
    all_critical = []
    for cp_type in ['minima', 'saddles', 'maxima']:
        all_critical.extend([v for v, _ in critical_points.get(cp_type, [])])
    
    if len(all_critical) <= 1:
        for v in all_critical:
            persistence[v] = float('inf')
        return persistence
    
    critical_vals = np.array([float(field[v]) for v in all_critical])
    
    for i in tqdm(
        range(len(all_critical)),
        desc="Persistence pass",
        unit="pt",
        disable=not show_progress,
    ):
        v = all_critical[i]
        fv = critical_vals[i]
        other_vals = np.concatenate([critical_vals[:i], critical_vals[i+1:]])
        if len(other_vals) > 0:
            min_diff = float(np.min(np.abs(other_vals - fv)))
            persistence[v] = min_diff
        else:
            persistence[v] = float('inf')
    
    return persistence


# -------------------------- geodesic clipping --------------------------- #
class _DSU:
    """Simple disjoint-set union (union-find) for clustering nearby saddles."""
    def __init__(self, items: Sequence[int]):
        self.parent = {int(x): int(x) for x in items}
        self.rank = {int(x): 0 for x in items}
    def find(self, x: int) -> int:
        x = int(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

def _edge_adjacency_with_weights(
    neighbors: List[Set[int]],
    verts: np.ndarray,
) -> List[List[Tuple[int, float]]]:
    """Build a weighted adjacency graph from the one-ring neighbourhoods.
    
    Parameters
    ----------
    neighbors : List[Set[int]]
        One-ring adjacency.
    verts : np.ndarray
        Vertex coordinates.
    
    Returns
    -------
    List[List[Tuple[int, float]]]
        Weighted adjacency list suitable for Dijkstra."""
    V = len(neighbors)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(V)]
    P = np.asarray(verts, dtype=float)
    for i in range(V):
        pi = P[i]
        lst = []
        for j in neighbors[i]:
            w = float(np.linalg.norm(P[int(j)] - pi))
            lst.append((int(j), w))
        adj[i] = lst
    return adj

def _dijkstra_with_cutoff(
    src: int,
    adj: List[List[Tuple[int, float]]],
    cutoff: float,
    csr_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Dict[int, float]:
    """Compute Dijkstra distances with an optional cutoff radius.
    
    Parameters
    ----------
    src : int
        Starting vertex index.
    adj : List[List[Tuple[int, float]]]
        Weighted adjacency list.
    cutoff : float
        Maximum distance to explore.
    csr_data : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        Optional CSR representation used when numba kernels are available.
    
    Returns
    -------
    Dict[int, float]
        Distance values truncated at `cutoff`."""
    if csr_data is not None and dijkstra_with_cutoff_jit is not None:
        indptr, indices, weights = csr_data
        dist = dijkstra_with_cutoff_jit(indptr, indices, weights, int(src), float(cutoff))
        if dist is not None:
            result: Dict[int, float] = {}
            for idx, value in enumerate(dist):
                if value <= cutoff:
                    result[int(idx)] = float(value)
            return result

    import heapq
    V = len(adj)
    INF = float("inf")
    dist = [INF] * V
    dist[src] = 0.0
    pq = [(0.0, int(src))]
    out: Dict[int, float] = {}
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if d > cutoff:
            break
        out[u] = d
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v] and nd <= cutoff:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return out

def _compute_local_saliency(
    v: int,
    field: np.ndarray,
    neighbors: List[Set[int]],
) -> float:
    """Measure local scalar-field saliency around a vertex via gradient heuristics.
    
    Parameters
    ----------
    v : int
        Vertex index.
    field : np.ndarray
        Scalar field sampled per vertex.
    neighbors : List[Set[int]]
        One-ring adjacency.
    
    Returns
    -------
    float
        Saliency score emphasising strong local contrast."""
    nbs = list(neighbors[v])
    if not nbs:
        return 0.0
    fv = float(field[v])
    diffs = [abs(float(field[u]) - fv) for u in nbs]
    if not diffs:
        return 0.0
    return float(np.median(diffs))

def clip_saddles_by_geodesic(
    faces: np.ndarray,
    vertices: np.ndarray,
    saddles: Sequence[int],
    *,
    threshold: float,
    scores: Optional[Dict[int, float]] = None,
    neighbors: Optional[List[Set[int]]] = None,
    show_progress: bool = True,
    field_for_saliency: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """Cluster saddle vertices by geodesic distance and keep the strongest representative per cluster.
    
    Parameters
    ----------
    faces : np.ndarray
        Triangle indices.
    vertices : np.ndarray
        Vertex coordinates.
    saddles : Sequence[int]
        Indices of candidate saddle vertices.
    threshold : float
        Geodesic distance threshold for clustering.
    scores : Dict[int, float], optional
        Per-vertex score used to choose the representative.
    neighbors : Optional[List[Set[int]]]
        One-ring adjacency.
    show_progress : bool, optional
        Display progress indicators.
    field_for_saliency : Optional[np.ndarray]
        Scalar field used when recomputing saliency-based scores.
    
    Returns
    -------
    Tuple[np.ndarray, Dict[int, List[int]]]
        Kept saddle indices and mapping from discarded saddles to their representative."""
    saddles_arr = np.array(sorted(set(int(x) for x in saddles)), dtype=int)
    if saddles_arr.size <= 1 or threshold <= 0 or not np.isfinite(threshold):
        return saddles_arr, {int(v): [int(v)] for v in saddles_arr}

    V = int(vertices.shape[0])
    F = np.asarray(faces, dtype=int)

    if neighbors is None:
        neighbors, _links = _build_vertex_ring_and_links(F, V, show_progress=show_progress)
    adj = _edge_adjacency_with_weights(neighbors, np.asarray(vertices, dtype=float))

    csr_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    if HAS_NUMBA and dijkstra_with_cutoff_jit is not None:
        indptr = np.zeros(V + 1, dtype=np.int64)
        total = 0
        for i in range(V):
            total += len(adj[i])
            indptr[i + 1] = total
        indices = np.empty(total, dtype=np.int64) if total else np.empty(0, dtype=np.int64)
        weights_arr = np.empty(total, dtype=np.float64) if total else np.empty(0, dtype=np.float64)
        ptr = 0
        for i in range(V):
            lst = adj[i]
            for nb, w in lst:
                indices[ptr] = int(nb)
                weights_arr[ptr] = float(w)
                ptr += 1
        csr_data = (indptr, indices, weights_arr)

    # Prepare scores
    score_map: Dict[int, float] = {}
    if scores is not None:
        score_map.update({int(k): float(v) for k, v in scores.items()})
    elif field_for_saliency is not None:
        for s in saddles_arr:
            score_map[int(s)] = _compute_local_saliency(int(s), np.asarray(field_for_saliency, dtype=float), neighbors)
    else:
        for s in saddles_arr:
            score_map[int(s)] = 0.0

    # Union saddles that lie within the geodesic radius
    dsu = _DSU(saddles_arr)
    saddle_set = set(int(x) for x in saddles_arr)

    iterator = saddles_arr
    if show_progress and saddles_arr.size > 20:
        iterator = tqdm(saddles_arr, desc="Clustering saddles (geodesic)", unit="seed", leave=False)

    for s in iterator:
        dmap = _dijkstra_with_cutoff(int(s), adj, float(threshold), csr_data=csr_data)
        # Connect with any other saddle found within the radius
        for t in dmap.keys():
            if t in saddle_set and t != int(s):
                dsu.union(int(s), int(t))

    # Collect clusters
    clusters: Dict[int, List[int]] = defaultdict(list)
    for s in saddles_arr:
        r = dsu.find(int(s))
        clusters[r].append(int(s))

    # Choose the best in each cluster
    kept: List[int] = []
    for r, members in clusters.items():
        # Pick max score; tie-break by smallest index
        best = max(members, key=lambda v: (float(score_map.get(int(v), 0.0)), -int(v)))
        kept.append(int(best))

    kept = np.array(sorted(set(kept)), dtype=int)
    # Remap clusters to use the chosen representative as key
    repmap: Dict[int, List[int]] = {}
    for r, members in clusters.items():
        rep = max(members, key=lambda v: (float(score_map.get(int(v), 0.0)), -int(v)))
        repmap[int(rep)] = sorted(set(members))

    return kept, repmap

# ----------------------------- classifier ----------------------------- #
def classify_critical_points(
    scalar_field: np.ndarray,
    faces: np.ndarray,
    *,
    vertices: Optional[np.ndarray] = None,
    clip_saddles_geodesic_threshold: float = 0.0,
    clip_saddles_strategy: str = 'confidence',
    eps: float = 1e-12,
    resolve_plateaus: bool = True,     # accepted for API compatibility (unused)
    normalize: bool = True,
    allow_odd_as_saddle: bool = False,
    smooth_field: bool = False,
    smooth_iterations: int = 2,
    smooth_weight: float = 0.5,
    adaptive_eps: bool = True,
    compute_confidence: bool = True,
    persistence_threshold: float = 0.0,
    progress: bool = True,
) -> Dict[str, np.ndarray]:
    """Classify vertices of a scalar field into minima, saddles, maxima, and regular points.
    
    Parameters
    ----------
    scalar_field : np.ndarray
        Scalar field sampled per vertex.
    faces : np.ndarray
        Triangle indices.
    vertices : Optional[np.ndarray]
        Vertex coordinates, required for certain heuristics.
    clip_saddles_geodesic_threshold : float, optional
        Distance threshold used when clustering near-duplicate saddles.
    clip_saddles_strategy : str, optional
        Strategy for ranking saddles when clustering.
    eps : float, optional
        Global epsilon tolerance for sign-change tests.
    smooth_field : bool, optional
        Enable Laplacian smoothing prior to classification.
    smooth_iterations : int, optional
        Iteration count used when `smooth_field` is `True`.
    adaptive_eps : bool, optional
        Enable per-vertex epsilon computation.
    compute_confidence : bool, optional
        Request per-vertex confidence scores.
    persistence_threshold : float, optional
        Filter critical points whose persistence falls below the threshold.
    progress : bool, optional
        Display progress indicators during processing.
    
    Returns
    -------
    Dict[str, np.ndarray]
        Classification results keyed by critical type. When requested, the dictionary
        also contains confidence scores and persistence values."""
    vals = np.asarray(scalar_field, dtype=float)
    V = int(vals.size)
    if V == 0:
        return {k: np.array([], dtype=int) for k in ('minima','saddles','maxima','regular')}

    if normalize:
        vmin = float(np.nanmin(vals)); vmax = float(np.nanmax(vals))
        rng = max(1e-16, vmax - vmin)
        f = (vals - vmin) / rng
    else:
        f = vals.copy()

    neighbors, link_pairs = _build_vertex_ring_and_links(faces, V, show_progress=progress)
    boundary_mask = _infer_boundary_mask_from_faces(faces, V, show_progress=progress)
    
    # Apply smoothing if requested
    if smooth_field:
        f = _laplacian_smooth_field(f, neighbors, boundary_mask, smooth_iterations, smooth_weight, show_progress=progress)

    neighbor_counts = np.array([len(neighbors[v]) for v in range(V)], dtype=np.int64)
    ring_lists: List[List[int]] = []
    ring_offsets = [0]
    ring_indices_flat: List[int] = []
    for v in range(V):
        ring = _ordered_one_ring(v, neighbors, link_pairs, faces if vertices is not None else None, vertices)
        ring_lists.append(ring)
        ring_offsets.append(ring_offsets[-1] + len(ring))
        ring_indices_flat.extend(ring)

    ring_offsets_arr = np.array(ring_offsets, dtype=np.int64)
    ring_indices_arr = np.array(ring_indices_flat, dtype=np.int64) if ring_indices_flat else np.empty(0, dtype=np.int64)
    boundary_uint8 = boundary_mask.astype(np.uint8)

    use_jit = HAS_NUMBA and classify_vertices_jit is not None
    eps_vec = np.full(V, float(eps), dtype=np.float64)

    if adaptive_eps:
        if use_jit and compute_adaptive_eps_jit is not None and ring_indices_arr.size > 0:
            eps_candidate = compute_adaptive_eps_jit(f, ring_offsets_arr, ring_indices_arr, float(eps))
            if eps_candidate is not None:
                eps_vec = eps_candidate
            else:
                use_jit = False
        if not use_jit:
            for v in range(V):
                eps_vec[v] = _compute_adaptive_eps(v, f, neighbors, eps)
    else:
        eps_vec.fill(float(eps))

    minima: List[Tuple[int, float]]
    maxima: List[Tuple[int, float]]
    saddles: List[Tuple[int, float]]
    regular: List[int]
    confidence_scores: Dict[int, float] = {}

    if use_jit:
        class_codes, confidences_arr = classify_vertices_jit(
            f,
            ring_offsets_arr,
            ring_indices_arr,
            boundary_uint8,
            eps_vec,
            neighbor_counts,
            bool(allow_odd_as_saddle),
            bool(compute_confidence),
        )
        if class_codes is not None and confidences_arr is not None:
            minima_idx = np.where(class_codes == 1)[0]
            maxima_idx = np.where(class_codes == 3)[0]
            saddles_idx = np.where(class_codes == 2)[0]
            regular_idx = np.where(class_codes == 0)[0]

            minima = [(int(v), float(confidences_arr[v])) for v in minima_idx]
            maxima = [(int(v), float(confidences_arr[v])) for v in maxima_idx]
            saddles = [(int(v), float(confidences_arr[v])) for v in saddles_idx]
            regular = [int(v) for v in regular_idx]

            if compute_confidence:
                crit_arrays = [arr for arr in (minima_idx, maxima_idx, saddles_idx) if arr.size]
                if crit_arrays:
                    for v in np.concatenate(crit_arrays):
                        confidence_scores[int(v)] = float(confidences_arr[v])
        else:
            use_jit = False

    if not use_jit:
        minima = []
        maxima = []
        saddles = []
        regular = []
        confidence_scores = {}

        pbar = tqdm(
            range(V),
            desc="Classifying critical points",
            unit="vtx",
            disable=not progress,
        )
        for v in pbar:
            if boundary_mask[v]:
                regular.append(v)
                continue

            ring = ring_lists[v]
            if len(ring) < 3:
                regular.append(v)
                continue

            eps_v = float(eps_vec[v]) if adaptive_eps else float(eps)

            diffs = [float(f[v] - f[u]) for u in ring]
            C = _count_sign_changes_closed(diffs, eps_v)

            confidence = 1.0
            if compute_confidence:
                confidence = _compute_confidence_score(v, f, neighbors, ring, C, eps_v)
                confidence_scores[v] = confidence

            if C == 0:
                pos = any(d > eps_v for d in diffs)
                neg = any(d < -eps_v for d in diffs)
                if pos and not neg:
                    maxima.append((v, confidence))
                elif neg and not pos:
                    minima.append((v, confidence))
                else:
                    regular.append(v)
            elif C == 2:
                regular.append(v)
            elif (C % 2 == 0 and C >= 4) or (allow_odd_as_saddle and C > 2):
                saddles.append((v, confidence))
            else:
                regular.append(v)

            if (v & 4095) == 0:
                pbar.set_postfix(min=len(minima), sad=len(saddles), max=len(maxima), refresh=False)
        pbar.close()

    # Apply persistence filtering if requested

    # Optional geodesic-based *clipping* of saddle points (to merge near-duplicates)
    if clip_saddles_geodesic_threshold and vertices is not None and len(saddles) > 1:
        # Build a score map according to the chosen strategy
        score_map: Dict[int, float] = {}
        if clip_saddles_strategy.lower() in ("confidence", "conf") and compute_confidence:
            for v, _c in saddles:
                score_map[int(v)] = float(confidence_scores.get(int(v), 0.0))
        elif clip_saddles_strategy.lower() in ("saliency", "contrast"):
            for v, _c in saddles:
                score_map[int(v)] = _compute_local_saliency(int(v), f, neighbors)
        # Run geodesic clustering
        kept_saddles, _cluster_map = clip_saddles_by_geodesic(
            faces=faces,
            vertices=np.asarray(vertices, dtype=float),
            saddles=[int(v) for v, _ in saddles],
            threshold=float(clip_saddles_geodesic_threshold),
            scores=score_map,
            neighbors=neighbors,
            show_progress=progress,
            field_for_saliency=f if clip_saddles_strategy.lower() in ("saliency", "contrast") else None,
        )
        kept_set = set(int(v) for v in kept_saddles)
        saddles = [(int(v), float(c)) for (v, c) in saddles if int(v) in kept_set]

    if persistence_threshold > 0:
        critical_dict = {'minima': minima, 'saddles': saddles, 'maxima': maxima}
        persistence = _compute_persistence(f, critical_dict, neighbors, show_progress=progress)
        
        minima = [(v, c) for v, c in minima if persistence.get(v, 0) >= persistence_threshold]
        saddles = [(v, c) for v, c in saddles if persistence.get(v, 0) >= persistence_threshold]
        maxima = [(v, c) for v, c in maxima if persistence.get(v, 0) >= persistence_threshold]
        
        # Move filtered points to regular
        all_kept = set([v for v, _ in minima + saddles + maxima])
        for v in range(V):
            if not boundary_mask[v] and v not in all_kept and v not in regular:
                regular.append(v)
    
    # Sort and extract indices (for backward compatibility)
    minima_idx = np.array(sorted(set(v for v, _ in minima)), dtype=int)
    saddles_idx = np.array(sorted(set(v for v, _ in saddles)), dtype=int)
    maxima_idx = np.array(sorted(set(v for v, _ in maxima)), dtype=int)
    regular_idx = np.array(sorted(set(regular)), dtype=int)
    
    result = {
        "minima": minima_idx,
        "saddles": saddles_idx,
        "maxima": maxima_idx,
        "regular": regular_idx,
    }
    
    if compute_confidence:
        result["confidence"] = confidence_scores
    
    return result
