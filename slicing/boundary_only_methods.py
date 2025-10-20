# -*- coding: utf-8 -*-
"""
Boundary-only slicing runners.

These methods generate scalar fields from geodesic distances to the mesh
boundaries ONLY (upper, lower, or a tween between both), deliberately
ignoring saddle points. Each runner returns a SlicingMethodResult and
uses the existing adaptive iso-slice extractor, so results are directly
compatible with the JSON exporter.

Maintainer: Abdallah Kamhawi (PhD researcher, DART Laboratory; Kamhawi@umich.edu)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from loaders.mesh_loader import MeshLoader
from slicing.slicing_base import SlicingBaseGraph
from slicing.config import get_slicer_defaults
from utilities.config_utils import (
    coerce_axis,
    coerce_bool,
    coerce_float,
    coerce_int,
)

# Local helpers -----------------------------------------------------------

def _load_default_configs() -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Return (general_cfg, boundaries_cfg, iso_cfg) from the slicer defaults.
    """
    settings = get_slicer_defaults()
    return (
        settings.get("general", {}),
        settings.get("boundaries", {}),
        settings.get("iso_slices", {}),
    )


def _initialise_graph(
    mesh_path: Path,
    show_progress: bool,
    general_cfg: Dict[str, Any],
    boundaries_cfg: Dict[str, Any],
) -> tuple[MeshLoader, SlicingBaseGraph, str]:
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

    return {
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


def _safe_min_distance(graph: SlicingBaseGraph, side: str) -> np.ndarray:
    """
    Get per-vertex geodesic distance to the closest boundary on the given side.
    """
    # Public helper exposed by SlicingBaseGraph; raises if side labels are missing.
    # See: compute_min_distance_to_boundary_side(...) in slicing_base.py
    return graph.compute_min_distance_to_boundary_side(side)


# Runners ----------------------------------------------------------------

def run_upper_only_method(mesh_path: Path, layer_height: float, show_progress: bool = True):
    """Slices derived from distance to the UPPER boundaries only (no saddles)."""
    # Local import to avoid circular dependency at module import time
    from slicing.methods import SlicingMethodResult  # type: ignore

    general_cfg, boundaries_cfg, iso_cfg = _load_default_configs()
    mesh, graph, _ = _initialise_graph(mesh_path, show_progress, general_cfg, boundaries_cfg)

    scalar_field = _safe_min_distance(graph, "upper")

    iso_kwargs = _build_iso_kwargs(iso_cfg, layer_height)
    slices = graph.extract_iso_slices(scalar_field=scalar_field, **iso_kwargs)

    metadata = {
        "field": "min_distance_to_upper",
        "requested_layer_height": float(layer_height),
    }
    return SlicingMethodResult(
        name="upper_only",
        display_name="Upper Boundary Only",
        graph=graph,
        slices=slices,
        layer_height=float(layer_height),
        metadata=metadata,
        scalar_field=scalar_field,
    )


def run_lower_only_method(mesh_path: Path, layer_height: float, show_progress: bool = True):
    """Slices derived from distance to the LOWER boundaries only (no saddles)."""
    from slicing.methods import SlicingMethodResult  # type: ignore

    general_cfg, boundaries_cfg, iso_cfg = _load_default_configs()
    mesh, graph, _ = _initialise_graph(mesh_path, show_progress, general_cfg, boundaries_cfg)

    scalar_field = _safe_min_distance(graph, "lower")

    iso_kwargs = _build_iso_kwargs(iso_cfg, layer_height)
    slices = graph.extract_iso_slices(scalar_field=scalar_field, **iso_kwargs)

    metadata = {
        "field": "min_distance_to_lower",
        "requested_layer_height": float(layer_height),
    }
    return SlicingMethodResult(
        name="lower_only",
        display_name="Lower Boundary Only",
        graph=graph,
        slices=slices,
        layer_height=float(layer_height),
        metadata=metadata,
        scalar_field=scalar_field,
    )


def run_between_boundaries_method(mesh_path: Path, layer_height: float, show_progress: bool = True):
    """
    Tween field between LOWER and UPPER sides that ignores saddle points.

    We build φ_lower and φ_upper as minimum geodesic distances to each side
    and then use:
        r = φ_lower / (φ_lower + φ_upper + ε)
    so that r≈0 on the LOWER side and r≈1 on the UPPER side.
    """
    from slicing.methods import SlicingMethodResult  # type: ignore

    general_cfg, boundaries_cfg, iso_cfg = _load_default_configs()
    mesh, graph, _ = _initialise_graph(mesh_path, show_progress, general_cfg, boundaries_cfg)

    d_lo = _safe_min_distance(graph, "lower").astype(float)
    d_up = _safe_min_distance(graph, "upper").astype(float)

    # Replace infinities to keep a stable ratio
    finite = np.isfinite(d_lo) | np.isfinite(d_up)
    if not np.any(finite):
        # Degenerate case: everything unreachable; fall back to zeros
        r = np.zeros_like(d_lo, dtype=float)
    else:
        max_finite = float(np.nanmax(np.concatenate([d_lo[finite], d_up[finite]])))
        big = 10.0 * (max_finite if np.isfinite(max_finite) and max_finite > 0 else 1.0)
        d_lo = np.where(np.isfinite(d_lo), d_lo, big)
        d_up = np.where(np.isfinite(d_up), d_up, big)
        eps = np.finfo(float).eps
        r = d_lo / (d_lo + d_up + eps)

    iso_kwargs = _build_iso_kwargs(iso_cfg, layer_height)
    slices = graph.extract_iso_slices(scalar_field=r, **iso_kwargs)

    metadata = {
        "field": "ratio(lower, upper)",
        "requested_layer_height": float(layer_height),
    }
    return SlicingMethodResult(
        name="between_boundaries",
        display_name="Between Boundaries (No Saddles)",
        graph=graph,
        slices=slices,
        layer_height=float(layer_height),
        metadata=metadata,
        scalar_field=r,
    )


__all__ = [
    "run_upper_only_method",
    "run_lower_only_method",
    "run_between_boundaries_method",
]
