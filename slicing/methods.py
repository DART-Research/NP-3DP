"""Reusable slicing method wrappers for benchmarking comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from loaders.mesh_loader import MeshLoader
from slicing.config import get_slicer_defaults
from slicing.iso_slice_collection import IsoSliceCollection
from slicing.slicing_base import SlicingBaseGraph
from utilities.config_utils import (
    coerce_axis,
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_sequence,
)
from utilities.mesh_utils import _median_edge_length


@dataclass
class SlicingMethodResult:
    """Container bundling graph, slices, and metadata for a slicing strategy."""

    name: str
    display_name: str
    graph: SlicingBaseGraph
    slices: IsoSliceCollection
    layer_height: float
    metadata: Dict[str, Any]
    scalar_field: Optional[np.ndarray] = None
    saddle_vertices: Optional[np.ndarray] = None


MethodRunner = Callable[[Path, float, bool], SlicingMethodResult]


def _load_default_configs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    settings = get_slicer_defaults()
    return (
        settings.get("general", {}),
        settings.get("boundaries", {}),
        settings.get("critical_points", {}),
        settings.get("scalar_field", {}),
        settings.get("iso_slices", {}),
    )


def _initialise_graph(
    mesh_path: Path,
    show_progress: bool,
    general_cfg: Dict[str, Any],
    boundaries_cfg: Dict[str, Any],
) -> Tuple[MeshLoader, SlicingBaseGraph, str]:
    mesh = MeshLoader(str(mesh_path))
    show_bar = bool(coerce_bool(general_cfg.get("show_progress"), True) and show_progress)
    graph = SlicingBaseGraph(mesh, show_progress=show_bar)

    axis_for_boundaries = coerce_axis(boundaries_cfg.get("axis"), "z")
    side_tol = coerce_float(boundaries_cfg.get("side_tol"), 0.05)
    assign_middle = coerce_bool(boundaries_cfg.get("assign_middle_to_nearest"), True)
    overwrite_boundaries = coerce_bool(boundaries_cfg.get("overwrite"), True)

    graph.label_lower_upper_boundaries(
        axis=axis_for_boundaries,
        side_tol=side_tol,
        assign_middle_to_nearest=assign_middle,
        overwrite=overwrite_boundaries,
    )
    graph.assign_nearest_lower_upper_boundaries(
        axis=axis_for_boundaries,
        side_tol=side_tol,
    )
    return mesh, graph, axis_for_boundaries


def _build_iso_kwargs(iso_cfg: Dict[str, Any], layer_height: float) -> Dict[str, Any]:
    dr_values = iso_cfg.get("dr_clip", (1e-4, 0.2))
    if isinstance(dr_values, (list, tuple)) and len(dr_values) == 2:
        dr_clip = (
            coerce_float(dr_values[0], 1e-4),
            coerce_float(dr_values[1], 0.2),
        )
    else:
        dr_clip = (1e-4, 0.2)

    kwargs: Dict[str, Any] = {
        "layer_height": float(layer_height),
        "periodic": coerce_bool(iso_cfg.get("periodic"), True),
        "degree": coerce_int(iso_cfg.get("degree"), 3),
        "samples": coerce_int(iso_cfg.get("samples"), 200),
        "verbose": coerce_bool(iso_cfg.get("verbose"), False),
        "dedupe_decimals": coerce_int(iso_cfg.get("dedupe_decimals"), 6),
        "edge_tol": coerce_float(iso_cfg.get("edge_tol"), 1e-6),
        "dr_clip": dr_clip,
        "max_levels": coerce_int(iso_cfg.get("max_levels"), 100000),
        "include_end": coerce_bool(iso_cfg.get("include_end"), True),
        "progress_bar_width": coerce_int(iso_cfg.get("progress_bar_width"), 80),
        "controller_blend": coerce_float(iso_cfg.get("controller_blend"), 0.5),
        "min_component_points": coerce_int(iso_cfg.get("min_component_points"), 3),
    }
    return kwargs


def _principal_axis_direction(mesh: MeshLoader) -> np.ndarray:
    verts = np.asarray(mesh.vertices, dtype=float)
    if verts.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0], dtype=float)

    centered = verts - verts.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    idx = int(np.argmax(vals))
    direction = vecs[:, idx]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    direction = direction / norm

    span = verts.max(axis=0) - verts.min(axis=0)
    if np.dot(direction, span) < 0.0:
        direction = -direction
    return direction.astype(float)


# ------------------------ Existing methods ------------------------------ #

def run_conformal_method(mesh_path: Path, layer_height: float, show_progress: bool = True) -> SlicingMethodResult:
    general_cfg, boundaries_cfg, critical_cfg, scalar_cfg, iso_cfg = _load_default_configs()
    mesh, graph, axis_for_boundaries = _initialise_graph(mesh_path, show_progress, general_cfg, boundaries_cfg)

    height_axis = coerce_axis(scalar_cfg.get("axis_for_boundary"), axis_for_boundaries)
    height_field = graph.compute_axis_distance(height_axis)
    median_edge = _median_edge_length(mesh)

    clip_threshold = critical_cfg.get("clip_saddles_geodesic_threshold")
    if clip_threshold is None:
        clip_spec = critical_cfg.get("clip_saddles", {})
        mode = str(clip_spec.get("mode", "median_edge_multiplier")).lower()
        value = coerce_float(clip_spec.get("value"), 1.5)
        clip_threshold = value * median_edge if mode == "median_edge_multiplier" else value
    clip_threshold = max(0.0, coerce_float(clip_threshold, 0.0))

    critical_kwargs: Dict[str, Any] = {
        "eps": coerce_float(critical_cfg.get("eps"), 1e-10),
        "clip_saddles_geodesic_threshold": clip_threshold,
        "clip_saddles_strategy": str(critical_cfg.get("clip_saddles_strategy", "confidence")),
        "resolve_plateaus": coerce_bool(critical_cfg.get("resolve_plateaus"), True),
        "normalize": coerce_bool(critical_cfg.get("normalize"), True),
        "smooth_field": coerce_bool(critical_cfg.get("smooth_field"), False),
        "smooth_iterations": coerce_int(critical_cfg.get("smooth_iterations"), 2),
        "adaptive_eps": coerce_bool(critical_cfg.get("adaptive_eps"), True),
        "compute_confidence": coerce_bool(critical_cfg.get("compute_confidence"), True),
        "persistence_threshold": coerce_float(critical_cfg.get("persistence_threshold"), 0.0),
    }

    saddles = graph.detect_saddle_points(
        height_field,
        annotate=coerce_bool(critical_cfg.get("annotate"), True),
        multi_scale=coerce_bool(critical_cfg.get("multi_scale"), False),
        scale_levels=coerce_int(critical_cfg.get("scale_levels"), 3),
        min_consensus=coerce_float(critical_cfg.get("min_consensus"), 0.6),
        **critical_kwargs,
    )

    scalar_kwargs: Dict[str, Any] = {
        "axis_for_boundary": height_axis,
        "n": coerce_float(scalar_cfg.get("n"), 2.0),
        "eps": coerce_float(scalar_cfg.get("eps"), 1e-9),
        "chunk_progress": coerce_bool(scalar_cfg.get("chunk_progress"), True),
    }
    radii_override = scalar_cfg.get("radii_per_saddle")
    radii_values = list(coerce_sequence(radii_override))
    if radii_values:
        scalar_kwargs["radii_per_saddle"] = radii_values

    scalar_field = graph.compute_conforming_scalar_field(
        saddle_vertices=saddles,
        **scalar_kwargs,
    )

    iso_kwargs = _build_iso_kwargs(iso_cfg, layer_height)
    slices = graph.extract_iso_slices(
        scalar_field=scalar_field,
        **iso_kwargs,
    )

    metadata = {
        "axis_for_boundaries": axis_for_boundaries,
        "height_axis": height_axis,
        "requested_layer_height": float(layer_height),
        "detected_saddles": int(len(saddles)),
    }
    return SlicingMethodResult(
        name="conformal",
        display_name="Conformal",
        graph=graph,
        slices=slices,
        layer_height=float(layer_height),
        metadata=metadata,
        scalar_field=scalar_field,
        saddle_vertices=np.asarray(saddles, dtype=int),
    )


def run_planar_z_method(mesh_path: Path, layer_height: float, show_progress: bool = True) -> SlicingMethodResult:
    general_cfg, boundaries_cfg, _critical_cfg, _scalar_cfg, iso_cfg = _load_default_configs()
    mesh, graph, _axis_for_boundaries = _initialise_graph(mesh_path, show_progress, general_cfg, boundaries_cfg)

    scalar_field = np.asarray(mesh.vertices, dtype=float)[:, 2]

    iso_kwargs = _build_iso_kwargs(iso_cfg, layer_height)
    slices = graph.extract_iso_slices(
        scalar_field=scalar_field,
        **iso_kwargs,
    )

    metadata = {
        "axis": "z",
        "requested_layer_height": float(layer_height),
    }
    return SlicingMethodResult(
        name="planar_z",
        display_name="Planar Z",
        graph=graph,
        slices=slices,
        layer_height=float(layer_height),
        metadata=metadata,
        scalar_field=scalar_field,
    )


def run_principal_axis_method(mesh_path: Path, layer_height: float, show_progress: bool = True) -> SlicingMethodResult:
    general_cfg, boundaries_cfg, _critical_cfg, _scalar_cfg, iso_cfg = _load_default_configs()
    mesh, graph, _axis_for_boundaries = _initialise_graph(mesh_path, show_progress, general_cfg, boundaries_cfg)

    direction = _principal_axis_direction(mesh)
    projected = np.asarray(mesh.vertices, dtype=float) @ direction

    iso_kwargs = _build_iso_kwargs(iso_cfg, layer_height)
    slices = graph.extract_iso_slices(
        scalar_field=projected,
        **iso_kwargs,
    )

    metadata = {
        "direction": direction.tolist(),
        "requested_layer_height": float(layer_height),
    }
    return SlicingMethodResult(
        name="principal_axis",
        display_name="Principal Axis",
        graph=graph,
        slices=slices,
        layer_height=float(layer_height),
        metadata=metadata,
        scalar_field=projected,
    )


# -------------------- Register additional methods ---------------------- #

# Boundary-only variants live in one file as requested.
try:
    from slicing.boundary_only_methods import (
        run_upper_only_method,
        run_lower_only_method,
        run_between_boundaries_method,
    )
except Exception as _err:
    # Defer import errors until call time for clearer diagnostics
    run_upper_only_method = None
    run_lower_only_method = None
    run_between_boundaries_method = None  # type: ignore

METHOD_REGISTRY: Dict[str, Tuple[str, MethodRunner]] = {
    "conformal": ("Conformal", run_conformal_method),
    "planar_z": ("Planar Z", run_planar_z_method),
    "principal_axis": ("Principal Axis", run_principal_axis_method),
    # New ones:
    "upper_only": ("Upper Boundary Only", run_upper_only_method),          # type: ignore[arg-type]
    "lower_only": ("Lower Boundary Only", run_lower_only_method),          # type: ignore[arg-type]
    "between_boundaries": ("Between Boundaries (No Saddles)", run_between_boundaries_method),  # type: ignore[arg-type]
}


def run_method_from_config(mesh_path: Path, layer_height: float, show_progress: bool = True) -> SlicingMethodResult:
    """
    Convenience dispatcher: read 'general.slicing_method' from config and run it.
    """
    settings = get_slicer_defaults()
    requested_key = str(settings.get("general", {}).get("slicing_method", "conformal")).lower()
    entry = METHOD_REGISTRY.get(requested_key)
    available = sorted(k for k, v in METHOD_REGISTRY.items() if v[1] is not None)
    if not entry or entry[1] is None:
        raise ValueError(
            f"Slicing method '{requested_key}' is not registered or unavailable. "
            f"Available methods: {', '.join(available)}"
        )

    _, runner = entry
    try:
        result = runner(mesh_path, layer_height, show_progress)
        result.metadata.setdefault("requested_method", requested_key)
        return result
    except ImportError as err:
        raise ImportError(
            f"Slicing method '{requested_key}' failed during execution: {err}"
        ) from err


__all__ = [
    "SlicingMethodResult",
    "MethodRunner",
    "run_conformal_method",
    "run_planar_z_method",
    "run_principal_axis_method",
    "run_method_from_config",
    "METHOD_REGISTRY",
]
