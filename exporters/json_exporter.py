"""JSON exporters for slicing outputs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:  # Optional acceleration when SciPy is available
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover - SciPy missing
    cKDTree = None  # type: ignore

from slicing.iso_slice import IsoSlice
from slicing.iso_slice_collection import IsoSliceCollection


def export_slice_collection_to_json(
    collection: IsoSliceCollection,
    output_dir: str | Path,
    *,
    filename: str = "slices.json",
    indent: int | None = 2,
    mesh: Any | None = None,
    graph: Any | None = None,
    saddle_vertices: Sequence[int] | np.ndarray | None = None,
) -> Path:
    """Serialise an :class:`~slicing.iso_slice_collection.IsoSliceCollection` to JSON.

    Parameters
    ----------
    collection
        Slices produced by the slicing pipeline.
    output_dir
        Target directory where the JSON export should be created.
    filename
        Name of the JSON file inside ``output_dir``.
    indent
        Indentation passed through to :func:`json.dumps` for readability.
    mesh
        Optional :class:`MeshLoader` instance providing vertex positions for
        topology annotations.
    graph
        Optional :class:`SlicingBaseGraph` used to query boundary metadata.
    saddle_vertices
        Optional iterable of vertex indices corresponding to saddle points.
    """
    if not isinstance(collection, IsoSliceCollection):
        raise TypeError("collection must be an IsoSliceCollection instance")

    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename

    resources = _prepare_topology_resources(mesh, graph, saddle_vertices)
    payload = _collection_to_payload(collection, resources)
    target.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def _collection_to_payload(
    collection: IsoSliceCollection,
    resources: Dict[str, Any],
) -> Dict[str, Any]:
    slices_payload = [
        _slice_to_payload(slice_obj, resources)
        for slice_obj in collection.slices
    ]

    meta: Dict[str, Any] = {
        "slice_count": len(slices_payload),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }

    if resources.get("saddles_meta"):
        meta["saddles"] = resources["saddles_meta"]
    if resources.get("mesh_meta"):
        meta["mesh"] = resources["mesh_meta"]

    return {
        "meta": meta,
        "slices": slices_payload,
    }


def _slice_to_payload(slice_obj: IsoSlice, resources: Dict[str, Any]) -> Dict[str, Any]:
    components = slice_obj.get_component_curves()
    lengths = getattr(slice_obj, "component_lengths", [])

    components_payload: List[Dict[str, Any]] = []
    for idx, component in enumerate(components):
        length = float(lengths[idx]) if idx < len(lengths) else float(slice_obj.total_length)
        topology = _build_component_topology(
            component,
            resources,
        )
        components_payload.append(
            {
                "length": length,
                "points": _array_to_lists(component),
                "topology": topology,
            }
        )

    return {
        "level": float(slice_obj.level),
        "layer_number": int(getattr(slice_obj, "layer_number", 0)),
        "total_length": float(getattr(slice_obj, "total_length", 0.0)),
        "component_count": int(getattr(slice_obj, "component_count", len(components_payload))),
        "curve": _array_to_lists(slice_obj.get_curve_points()),
        "components": components_payload,
        "raw_points": [list(pt) for pt in slice_obj.points],
    }


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------


def _prepare_topology_resources(
    mesh: Any | None,
    graph: Any | None,
    saddle_vertices: Sequence[int] | np.ndarray | None,
) -> Dict[str, Any]:
    resources: Dict[str, Any] = {}

    if mesh is not None:
        vertices = np.asarray(getattr(mesh, "vertices", []), dtype=float)
        if vertices.size:
            resources["mesh_vertices"] = vertices
            resources["mesh_meta"] = {
                "vertex_count": int(vertices.shape[0]),
            }
            if cKDTree is not None:
                resources["mesh_tree"] = cKDTree(vertices)
    if graph is not None:
        resources["graph"] = graph

    if saddle_vertices is not None and "mesh_vertices" in resources:
        saddle_idx = np.asarray(list(saddle_vertices), dtype=int)
        saddle_idx = saddle_idx[np.isfinite(saddle_idx)]
        saddle_idx = saddle_idx[(saddle_idx >= 0) & (saddle_idx < resources["mesh_vertices"].shape[0])]
        if saddle_idx.size:
            resources["saddle_vertices"] = saddle_idx
            resources["saddle_points"] = resources["mesh_vertices"][saddle_idx]
            if cKDTree is not None:
                resources["saddle_tree"] = cKDTree(resources["saddle_points"])
            resources["saddles_meta"] = [
                {
                    "index": int(idx),
                    "point": _vector_to_list(resources["mesh_vertices"][idx]),
                }
                for idx in saddle_idx
            ]
    return resources


def _build_component_topology(component: np.ndarray, resources: Dict[str, Any]) -> Dict[str, Any]:
    points = np.asarray(component, dtype=float)
    if points.size == 0:
        return {
            "nearest_vertex": None,
            "boundary_side": None,
            "boundary_number": None,
            "dominant_saddle": None,
            "saddle_signature": [],
            "saddle_histogram": [],
        }

    mesh_vertices: Optional[np.ndarray] = resources.get("mesh_vertices")
    mesh_tree = resources.get("mesh_tree")
    graph = resources.get("graph")
    saddle_vertices = resources.get("saddle_vertices")
    saddle_points = resources.get("saddle_points")

    centroid = points.mean(axis=0)
    nearest_vertex: Optional[int] = None
    vertex_distance: Optional[float] = None

    if mesh_vertices is not None and mesh_vertices.size:
        if mesh_tree is not None:
            vertex_distance, idx = mesh_tree.query(centroid)
            nearest_vertex = int(idx)
            vertex_distance = float(vertex_distance)
        else:
            diffs = mesh_vertices - centroid
            dist_sq = np.einsum("ij,ij->i", diffs, diffs)
            idx = int(np.argmin(dist_sq))
            nearest_vertex = idx
            vertex_distance = float(np.sqrt(dist_sq[idx]))

    boundary_side = None
    boundary_number = None
    critical_type = None
    if graph is not None and nearest_vertex is not None:
        G = getattr(graph, "G", None)
        if G is not None and G.has_node(nearest_vertex):
            data = G.nodes[nearest_vertex]
            boundary_side = data.get("boundary_side")
            boundary_number = data.get("boundary_number")
            critical_type = data.get("critical_type")

    saddle_signature: List[int] = []
    saddle_histogram: List[Dict[str, Any]] = []
    dominant_saddle: Optional[int] = None

    if saddle_vertices is not None and saddle_points is not None and saddle_points.size and points.shape[0] > 0:
        # Compute nearest saddle for each point
        diffs = points[:, None, :] - saddle_points[None, :, :]
        dist_sq = np.einsum("ijk,ijk->ij", diffs, diffs)
        nearest_idx = np.argmin(dist_sq, axis=1)
        nearest_dist = np.sqrt(dist_sq[np.arange(dist_sq.shape[0]), nearest_idx])

        count = Counter(nearest_idx.tolist())
        distance_accum: Dict[int, List[float]] = defaultdict(list)
        for idx_val, dist_val in zip(nearest_idx, nearest_dist):
            distance_accum[int(idx_val)].append(float(dist_val))

        ordered = sorted(count.items(), key=lambda item: (-item[1], item[0]))
        saddle_signature = [int(saddle_vertices[idx]) for idx, _ in ordered[:3]]
        if ordered:
            dominant_saddle = int(saddle_vertices[ordered[0][0]])
        total_pts = float(points.shape[0])
        for idx_val, freq in ordered:
            distances = distance_accum[idx_val]
            saddle_histogram.append(
                {
                    "index": int(saddle_vertices[idx_val]),
                    "count": int(freq),
                    "fraction": float(freq / total_pts),
                    "mean_distance": float(sum(distances) / len(distances)),
                }
            )

    topology = {
        "nearest_vertex": int(nearest_vertex) if nearest_vertex is not None else None,
        "nearest_vertex_distance": vertex_distance,
        "boundary_side": boundary_side,
        "boundary_number": int(boundary_number) if boundary_number is not None else None,
        "critical_type": critical_type,
        "dominant_saddle": dominant_saddle,
        "saddle_signature": saddle_signature,
        "saddle_histogram": saddle_histogram,
    }
    return topology


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _array_to_lists(arr: np.ndarray) -> List[List[float]]:
    if not isinstance(arr, np.ndarray):
        return [list(map(float, point)) for point in arr]
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return []
    return arr.tolist()


def _vector_to_list(vec: np.ndarray) -> List[float]:
    return [float(v) for v in np.asarray(vec, dtype=float).tolist()]
