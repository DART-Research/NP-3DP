"""Assemble topology-aware iso-slice components from raw intersection segments.

Maintainer: Abdallah Kamhawi (PhD researcher, DART Laboratory; Kamhawi@umich.edu)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

__all__ = ["SliceComponent", "build_slice_components"]


@dataclass
class SliceComponent:

    """Structured representation of a single connected iso-curve component."""

    level: float
    component_id: int
    points: np.ndarray
    is_closed: bool
    curve_length: float = 0.0
    parent_component: int | None = None

    def __post_init__(self) -> None:
        """Compute the Euclidean arc length of ``points``."""
        self.points = np.asarray(self.points, dtype=float)
        if self.points.shape[0] < 2:
            self.curve_length = 0.0
            return
        diffs = np.diff(self.points, axis=0)
        length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        if self.is_closed and self.points.shape[0] > 2:
            length += float(np.linalg.norm(self.points[-1] - self.points[0]))
        self.curve_length = length


def build_slice_components(
    level: float,
    segments: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    min_points: int = 3,
    rounding_decimals: int = 6,
    connectivity_factor: float = 0.1,
    orientation_hint: np.ndarray | None = None,
) -> List[SliceComponent]:
    """Group raw intersection segments into ordered slice components.

    Parameters
    ----------
    level : float
        Normalised iso-value associated with the segments.
    segments : Sequence[Tuple[np.ndarray, np.ndarray]]
        Collection of raw intersection segments expressed as ``(p0, p1)``
        pairs of 3D coordinates.
    min_points : int, optional
        Minimum number of vertices required for a component to be retained.
    rounding_decimals : int, optional
        Number of decimals used when hashing points while ordering segments.
    connectivity_factor : float, optional
        Fraction of the median segment length used to derive the spatial
        threshold when stitching segments together. The value is multiplied
        with the characteristic length to obtain the cut-off used by the
        KD-tree based clustering.
    orientation_hint : np.ndarray, optional
        Preferred positive direction for the curve tangents / normals. When
        provided the connected components are flipped, if necessary, so that
        their orientation aligns with the hint.

    Returns
    -------
    List[SliceComponent]
        Distinct connected iso-curve components ordered by discovery.
    """

    segment_points: List[Tuple[np.ndarray, np.ndarray]] = []
    for seg in segments:
        if len(seg) != 2:
            raise ValueError("Segments must be provided as (p0, p1) tuples")
        p0 = np.asarray(seg[0], dtype=float)
        p1 = np.asarray(seg[1], dtype=float)
        if p0.shape != (3,) or p1.shape != (3,):
            raise ValueError("Segment endpoints must be 3D points")
        segment_points.append((p0, p1))

    if not segment_points:
        return []

    if orientation_hint is not None:
        orientation_hint = np.asarray(orientation_hint, dtype=float)

    grouped = _group_segments_by_connectivity(
        segment_points,
        rounding_decimals=rounding_decimals,
        connectivity_factor=connectivity_factor,
    )

    components: List[SliceComponent] = []
    for comp_id, comp_segments in enumerate(grouped):
        curve_points, is_closed = _order_component_segments(
            comp_segments,
            rounding_decimals=rounding_decimals,
        )
        if curve_points is None or curve_points.shape[0] < min_points:
            continue
        if is_closed and curve_points.shape[0] >= 2:
            if not np.allclose(curve_points[0], curve_points[-1]):
                curve_points = np.vstack([curve_points, curve_points[0]])
        if orientation_hint is not None and curve_points.shape[0] >= 2:
            curve_points = _ensure_curve_orientation(
                curve_points,
                orientation_hint,
                is_closed=is_closed,
            )
        components.append(
            SliceComponent(
                level=level,
                component_id=comp_id,
                points=curve_points,
                is_closed=is_closed,
            )
        )
    return components


def _group_segments_by_connectivity(
    segments: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    rounding_decimals: int,
    connectivity_factor: float,
) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    """Cluster segments into connectivity-aware groups using a KD-tree.

    Parameters
    ----------
    segments : Sequence[Tuple[np.ndarray, np.ndarray]]
        Collection of intersection segments.
    rounding_decimals : int
        Decimal precision used when hashing endpoints for exact matches.
    connectivity_factor : float
        Fraction of the median segment length used to derive the KD-tree
        radius.

    Returns
    -------
    List[List[Tuple[np.ndarray, np.ndarray]]]
        Segment groups, each corresponding to a connected component.
    """

    graph = nx.Graph()
    endpoints: List[np.ndarray] = []
    endpoint_to_segment: defaultdict[int, List[int]] = defaultdict(list)

    for idx, (p0, p1) in enumerate(segments):
        graph.add_node(idx)
        endpoints.append(p0)
        endpoint_to_segment[len(endpoints) - 1].append(idx)
        endpoints.append(p1)
        endpoint_to_segment[len(endpoints) - 1].append(idx)

    if not endpoints:
        return []

    edge_lengths = [
        np.linalg.norm(p1 - p0)
        for p0, p1 in segments
        if not np.allclose(p0, p1)
    ]
    if edge_lengths:
        typical_length = float(np.median(edge_lengths))
        connect_threshold = max(1e-6, typical_length * connectivity_factor)
    else:
        connect_threshold = 1e-3

    kdtree = KDTree(np.array(endpoints))
    for ep_idx_a, ep_idx_b in kdtree.query_pairs(connect_threshold):
        for seg_a in endpoint_to_segment[ep_idx_a]:
            for seg_b in endpoint_to_segment[ep_idx_b]:
                if seg_a != seg_b:
                    graph.add_edge(seg_a, seg_b)

    buckets: defaultdict[Tuple[float, float, float], List[int]] = defaultdict(list)
    for seg_idx, (p0, p1) in enumerate(segments):
        buckets[tuple(np.round(p0, rounding_decimals))].append(seg_idx)
        buckets[tuple(np.round(p1, rounding_decimals))].append(seg_idx)
    for members in buckets.values():
        if len(members) <= 1:
            continue
        first = members[0]
        for other in members[1:]:
            graph.add_edge(first, other)

    connected: List[List[Tuple[np.ndarray, np.ndarray]]] = []
    for component in nx.connected_components(graph):
        connected.append([segments[idx] for idx in component])
    return connected


def _order_component_segments(
    segments: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    rounding_decimals: int,
) -> Tuple[np.ndarray | None, bool]:
    """Order segments in a component to form a continuous polyline.

    Parameters
    ----------
    segments : Sequence
        Segments belonging to the same connectivity group.
    rounding_decimals : int
        Decimal precision used when collapsing endpoints onto grid points.

    Returns
    -------
    Tuple[np.ndarray | None, bool]
        Pair of ``(ordered_points, is_closed)``. ``ordered_points`` becomes
        ``None`` if the ordering fails due to insufficient data.
    """

    if not segments:
        return None, False

    adjacency: defaultdict[Tuple[float, float, float], List[Tuple[float, float, float]]] = defaultdict(list)
    for p0, p1 in segments:
        key0 = tuple(np.round(p0, rounding_decimals))
        key1 = tuple(np.round(p1, rounding_decimals))
        adjacency[key0].append(key1)
        adjacency[key1].append(key0)

    degrees = {node: len(neighbors) for node, neighbors in adjacency.items()}
    endpoints = [node for node, degree in degrees.items() if degree == 1]
    is_closed = len(endpoints) == 0
    start = endpoints[0] if endpoints else next(iter(adjacency))

    ordered: List[np.ndarray] = []
    visited: set[Tuple[float, float, float]] = set()
    current = start
    ordered.append(np.array(current, dtype=float))
    visited.add(current)

    while True:
        next_node = None
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                next_node = neighbor
                break
        if next_node is None:
            break
        ordered.append(np.array(next_node, dtype=float))
        visited.add(next_node)
        current = next_node

    if len(ordered) < 2:
        return None, False
    return np.vstack(ordered), is_closed



def _ensure_curve_orientation(
    points: np.ndarray,
    hint: np.ndarray,
    *,
    is_closed: bool,
) -> np.ndarray:
    """Return ``points`` with orientation chosen to align with ``hint``."""
    hint = np.asarray(hint, dtype=float)
    if not np.all(np.isfinite(hint)):
        return np.asarray(points, dtype=float)
    norm = float(np.linalg.norm(hint))
    if norm < 1e-9:
        return np.asarray(points, dtype=float)
    direction = hint / norm

    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return pts.copy()

    if is_closed:
        area_vec = _newell_area_vector(pts)
        area_norm = float(np.linalg.norm(area_vec))
        if area_norm < 1e-9:
            area_vec = _best_fit_normal(pts)
            area_norm = float(np.linalg.norm(area_vec))
        if area_norm < 1e-9:
            return pts.copy()
        area_dir = area_vec / area_norm
        if np.dot(area_dir, direction) < 0.0:
            return pts[::-1].copy()
        return pts.copy()

    delta = pts[-1] - pts[0]
    if float(np.linalg.norm(delta)) < 1e-9:
        normal = _best_fit_normal(pts)
        if float(np.linalg.norm(normal)) < 1e-9:
            return pts.copy()
        if np.dot(normal, direction) < 0.0:
            return pts[::-1].copy()
        return pts.copy()

    if np.dot(delta, direction) < 0.0:
        return pts[::-1].copy()
    return pts.copy()


def _newell_area_vector(points: np.ndarray) -> np.ndarray:
    """Compute an area-weighted normal using Newell's method."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] > 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    total = np.zeros(3, dtype=float)
    n = pts.shape[0]
    if n < 3:
        return total
    for i in range(n):
        p = pts[i]
        q = pts[(i + 1) % n]
        total += np.cross(p, q)
    return total * 0.5


def _best_fit_normal(points: np.ndarray) -> np.ndarray:
    """Return the smallest-variance principal axis of ``points``."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return np.zeros(3, dtype=float)
    centered = pts - pts.mean(axis=0)
    cov = centered.T @ centered
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return np.zeros(3, dtype=float)
    idx = int(np.argmin(vals))
    normal = vecs[:, idx]
    return normal.astype(float)
