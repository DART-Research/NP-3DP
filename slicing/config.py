"""Slicer configuration helpers with YAML override support.

Maintainer: Abdallah Kamhawi (PhD researcher, DART Laboratory; Kamhawi@umich.edu)
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional




_DEFAULTS: Dict[str, Any] = {
    "general": {
        "show_progress": True,
        # NEW: choose which slicing method to run by default.
        # Options (see slicing.methods.METHOD_REGISTRY):
        #   'conformal', 'planar_z', 'principal_axis',
        #   'upper_only', 'lower_only', 'between_boundaries'
        "slicing_method": "conformal",
    },
    "boundaries": {
        "axis": "z",
        "side_tol": 0.05,
        "assign_middle_to_nearest": True,
        "overwrite": True,
    },
    "critical_points": {
        "annotate": True,
        "compute_confidence": True,
        "clip_saddles_geodesic_threshold": None,
        "clip_saddles": {
            "mode": "median_edge_multiplier",
            "value": 1.5,
        },
        "clip_saddles_strategy": "confidence",
        "multi_scale": False,
        "scale_levels": 3,
        "min_consensus": 0.6,
        "eps": 1e-10,
        "resolve_plateaus": True,
        "normalize": True,
        "smooth_field": False,
        "smooth_iterations": 2,
        "adaptive_eps": True,
        "persistence_threshold": 0.0,
    },
    "scalar_field": {
        "axis_for_boundary": "z",
        "radii_per_saddle": None,
        "n": 2.0,
        "eps": 1e-9,
        "chunk_progress": True,
    },
    "iso_slices": {
        "layer_height": 15,
        "periodic": True,
        "degree": 3,
        "samples": 1800,
        "verbose": False,
        "dedupe_decimals": 6,
        "edge_tol": 1e-6,
        "dr_clip": [1e-4, 0.2],
        "max_levels": 100000,
        "include_end": True,
        "progress_bar_width": 80,
        "controller_blend": 0.5,
        "min_component_points": 3,
    },
    "exports": {
        "slice_export": True,
        "export_directory": "export_output/slices",
    },
}

_EFFECTIVE_DEFAULTS: Dict[str, Any] = copy.deepcopy(_DEFAULTS)


def _deep_update(destination: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Recursively merge ``source`` into ``destination`` in-place."""
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            _deep_update(destination[key], value)
        else:
            destination[key] = value


def _candidate_paths(path: Optional[str]) -> Iterable[Path]:
    """Yield the YAML file locations that are probed for overrides."""
    if path:
        yield Path(path)
        return

    env_override = os.environ.get("SLICER_DEFAULTS_YAML")
    if env_override:
        yield Path(env_override)

    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parent

    yield project_root / "slicer_defaults.yaml"
    yield Path.cwd() / "slicer_defaults.yaml"
    yield package_dir / "slicer_defaults.yaml"


def reload_slicer_defaults(path: Optional[str] = None) -> None:
    defaults = copy.deepcopy(_DEFAULTS)


    global _EFFECTIVE_DEFAULTS
    _EFFECTIVE_DEFAULTS = defaults


def get_slicer_defaults() -> Dict[str, Any]:
    """Return a deep copy of all effective configuration groups."""
    return copy.deepcopy(_EFFECTIVE_DEFAULTS)


def get_slicer_section(section: str) -> Dict[str, Any]:
    """Return a deep copy of the configuration subset named ``section``."""
    defaults = get_slicer_defaults()
    section_defaults = defaults.get(section, {})
    if isinstance(section_defaults, dict):
        return copy.deepcopy(section_defaults)
    return {}


reload_slicer_defaults()
