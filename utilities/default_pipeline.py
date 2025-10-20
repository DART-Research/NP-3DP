"""Shared helpers for reproducing the default slicing pipeline."""

from __future__ import annotations

from pathlib import Path
from exporters import export_slice_collection_to_json
from typing import Tuple

from slicing.config import get_slicer_defaults
from slicing.iso_slice_collection import IsoSliceCollection
from slicing.slicing_base import SlicingBaseGraph
from loaders.mesh_loader import MeshLoader
from utilities.mesh_utils import _median_edge_length
from utilities.config_utils import (
    coerce_axis,
    coerce_bool,
    coerce_float,
    coerce_int,
    coerce_sequence,
)


__all__ = ["extract_iso_slices_from_defaults"]


def extract_iso_slices_from_defaults(
    *, return_mesh: bool = False
) -> IsoSliceCollection | Tuple[IsoSliceCollection, MeshLoader]:
    """Replicate the demo pipeline to generate iso-slices using defaults."""
    settings = get_slicer_defaults()
    general_cfg = settings.get("general", {})
    boundaries_cfg = settings.get("boundaries", {})
    critical_cfg = settings.get("critical_points", {})
    scalar_cfg = settings.get("scalar_field", {})
    iso_cfg = settings.get("iso_slices", {})
    exports_cfg = settings.get("exports", {})

    mesh_path = Path("assets") / "beam_3b.obj"
    print(f"Loading mesh from {mesh_path} ...")
    mesh = MeshLoader(str(mesh_path))

    show_progress = coerce_bool(general_cfg.get("show_progress"), True)

    print("Building slicing graph ...")
    sbg = SlicingBaseGraph(mesh, show_progress=show_progress)

    axis_for_boundaries = coerce_axis(boundaries_cfg.get("axis"), "z")
    side_tol = coerce_float(boundaries_cfg.get("side_tol"), 0.05)
    assign_middle = coerce_bool(boundaries_cfg.get("assign_middle_to_nearest"), True)
    overwrite_boundaries = coerce_bool(boundaries_cfg.get("overwrite"), True)

    sbg.label_lower_upper_boundaries(
        axis=axis_for_boundaries,
        side_tol=side_tol,
        assign_middle_to_nearest=assign_middle,
        overwrite=overwrite_boundaries,
    )
    sbg.assign_nearest_lower_upper_boundaries(
        axis=axis_for_boundaries,
        side_tol=side_tol,
    )

    print("Computing saddle annotations ...")
    height_axis = coerce_axis(scalar_cfg.get("axis_for_boundary"), axis_for_boundaries)
    height_field = sbg.compute_axis_distance(height_axis)
    median_edge = _median_edge_length(mesh)

    clip_threshold = critical_cfg.get("clip_saddles_geodesic_threshold")
    if clip_threshold is None:
        clip_spec = critical_cfg.get("clip_saddles", {})
        mode = str(clip_spec.get("mode", "median_edge_multiplier")).lower()
        value = coerce_float(clip_spec.get("value"), 1.5)
        if mode == "median_edge_multiplier":
            clip_threshold = value * median_edge
        else:
            clip_threshold = value
    clip_threshold = max(0.0, coerce_float(clip_threshold, 0.0))

    critical_kwargs = {
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

    saddles = sbg.detect_saddle_points(
        height_field,
        annotate=coerce_bool(critical_cfg.get("annotate"), True),
        multi_scale=coerce_bool(critical_cfg.get("multi_scale"), False),
        scale_levels=coerce_int(critical_cfg.get("scale_levels"), 3),
        min_consensus=coerce_float(critical_cfg.get("min_consensus"), 0.6),
        **critical_kwargs,
    )

    print("Constructing conforming scalar field ...")
    scalar_kwargs = {
        "axis_for_boundary": height_axis,
        "n": coerce_float(scalar_cfg.get("n"), 2.0),
        "eps": coerce_float(scalar_cfg.get("eps"), 1e-9),
        "chunk_progress": coerce_bool(scalar_cfg.get("chunk_progress"), True),
    }
    radii_override = scalar_cfg.get("radii_per_saddle")
    radii_values = list(coerce_sequence(radii_override))
    if radii_values:
        scalar_kwargs["radii_per_saddle"] = radii_values

    scalar_field = sbg.compute_conforming_scalar_field(
        saddle_vertices=saddles,
        **scalar_kwargs,
    )

    print("Extracting iso-slices with multi-component support ...")
    dr_clip_values = iso_cfg.get("dr_clip", (1e-4, 0.2))
    if isinstance(dr_clip_values, (list, tuple)) and len(dr_clip_values) == 2:
        dr_clip = (
            coerce_float(dr_clip_values[0], 1e-4),
            coerce_float(dr_clip_values[1], 0.2),
        )
    else:
        dr_clip = (1e-4, 0.2)

    iso_kwargs = {
        "layer_height": coerce_float(iso_cfg.get("layer_height"), 10.0),
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

    slices = sbg.extract_iso_slices(
        scalar_field=scalar_field,
        **iso_kwargs,
    )

    export_slices = coerce_bool(exports_cfg.get("slice_export"), False)
    if export_slices:
        export_dir_raw = exports_cfg.get("export_directory") or "analysis_output/slices"
        export_dir = Path(export_dir_raw)
        export_slice_collection_to_json(
            slices,
            export_dir,
            mesh=mesh,
            graph=sbg,
            saddle_vertices=saddles,
        )

    if return_mesh:
        return slices, mesh
    return slices
