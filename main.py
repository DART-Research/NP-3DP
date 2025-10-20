"""
Demonstration entry point for the iso-slicing pipeline.

The script loads a sample mesh, constructs the slicing graph, computes a
geometry-conforming scalar field, extracts multi-component iso-slices, and then
produces a handful of visual diagnostics. The goal is to provide a clear,
documented walkthrough that mirrors the typical workflow for downstream tools.

Maintainer: Abdallah Kamhawi (PhD researcher, DART Laboratory; Kamhawi@umich.edu)
"""
  
from __future__ import annotations
from pathlib import Path

from exporters import export_slice_collection_to_json
from slicing.config import get_slicer_defaults
from slicing.methods import run_method_from_config

from utilities.config_utils import (
    coerce_bool,
    coerce_float,
)
from utilities.iso_slice_collection_utils import summarise_components

from viz.slicing_base_viz import *
from viz.iso_slice_collection_viz import *


def main() -> None:
    """Execute the slicing pipeline on the demo mesh and emit visual reports."""
    settings = get_slicer_defaults()
    general_cfg = settings.get("general", {})
    iso_cfg = settings.get("iso_slices", {})
    exports_cfg = settings.get("exports", {})

    mesh_path = Path("assets") / "beam_3b.obj"
    print(f"Loading mesh from {mesh_path} ...")
    layer_height = coerce_float(iso_cfg.get("layer_height"), 10.0)
    show_progress = coerce_bool(general_cfg.get("show_progress"), True)

    method_key = str(general_cfg.get("slicing_method", "conformal")).lower()
    print(f"Selected slicing method from config: '{method_key}'")

    print("Building slicing graph and extracting iso-slices ...")
    result = run_method_from_config(mesh_path, layer_height, show_progress=show_progress)

    sbg = result.graph
    mesh = sbg.mesh
    slices = result.slices
    scalar_field = result.scalar_field
    saddle_vertices = result.saddle_vertices

    print(f"Resolved slicing method: {result.display_name} (key '{result.name}')")

    stats = summarise_components(slices)

    print(
        "Iso-slice summary: "
        f"{stats['total_slices']} slices, "
        f"{stats['bifurcation_slices']} with bifurcations, "
        f"up to {stats['max_components']} components"
    )

    if stats["bifurcation_levels"]:
        first = min(stats["bifurcation_levels"])
        last = max(stats["bifurcation_levels"])
        print(f"Bifurcations span levels [{first:.4f}, {last:.4f}]")

    export_slices = coerce_bool(exports_cfg.get("slice_export"), False)

    # Build descriptive export filename
    mesh_name = Path(mesh_path).stem
    layer_height_value = float(result.layer_height)
    method_name = result.name
    file_name = f"{mesh_name}_lh{layer_height_value:g}_{method_name}.json"
    print(f"Export filename will be: {file_name}")

    if export_slices:
        export_dir_raw = exports_cfg.get("export_directory") or "analysis_output/slices"
        export_dir = Path(export_dir_raw)
        try:
            output_file = export_slice_collection_to_json(
                slices,
                export_dir,
                filename=file_name,
                mesh=mesh,
                graph=sbg,
                saddle_vertices=saddle_vertices,
            )
            print(f"Slice export written -> {output_file}")
        except Exception as err:
            print(f"[export] Failed to write slice JSON: {err}")

    # print("Generating visualisations ...")
    # if scalar_field is not None:
    #     visualize_scalar_field(
    #         mesh,
    #         scalar_field,
    #         normalize=True,
    #     )
    # else:
    #     print("[viz] Skipping scalar field mesh visualisation (no scalar field supplied).")
    # visualize_slicing_graph(sbg)
    # visualize_slices(slices)
    # visualize_scalar_field_lower_only(sbg)
    # visualize_scalar_field_upper_only(sbg)
    # if saddle_vertices is not None and getattr(saddle_vertices, "size", len(saddle_vertices)) > 0:
    #     visualize_scalar_field_saddles_only(sbg)
    # else:
    #     print("[viz] Skipping saddle-only visualisation (no saddle annotations).")

    # print("Visualisation complete: scalar_field.html, graph_structure.html, slices.html")


if __name__ == "__main__":
    main()
