"""Central configuration helpers for visualization modules.

Maintainer: Abdallah Kamhawi (PhD researcher, DART Laboratory; Kamhawi@umich.edu)
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml is optional
    yaml = None


_DEFAULTS: Dict[str, Any] = {
    "global": {
        "save_html": False,
        "auto_open": True,
        "camera": {
            "eye": {"x": 1.5, "y": 1.5, "z": 1.5},
        },
        "hide_axes": True,
        "show_grid": False,
        "aspectmode": "data",
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
    },
    "slicing_base": {
        "colormap": "Turbo",
        "opacity": 1.0,
        "normalize": False,
        "node_size": 4,
        "show_internal_edges": True,
        "internal_edge_alpha": 0.25,
        "filepaths": {
            "scalar": "scalar_field.html",
            "graph": "slicing_graph.html",
            "upper": "scalar_upper.html",
            "lower": "scalar_lower.html",
            "saddles": "scalar_saddles.html",
        },
    },
    "iso_slices": {
        "colorscale": "Turbo",
        "line_width": 2.5,
        "show_colorbar": True,
        "title": "Iso-slices by Level",
        "filepaths": {
            "slices": "iso_curves.html",
        },
    },
    "volumetric_cells": {
        "colorscale": "Turbo",
        "opacity": 0.6,
        "show_edges": True,
        "edge_width": 1.5,
        "edge_color": "#303030",
        "save_html": False,
        "auto_open": True,
        "title": "Volumetric Cells",
        "filepaths": {
            "cells": "volumetric_cells.html",
            "mesh": "volumetric_cells_mesh.html",
            "structure": "volumetric_cells_structure.html",
        },
        "mesh": {
            "colorscale": "Turbo",
            "opacity": 1.0,
            "method": "nearest",
            "title": "Volumetric Metric on Mesh",
        },
    },
}

_EFFECTIVE_DEFAULTS: Dict[str, Any] = copy.deepcopy(_DEFAULTS)


def _deep_update(destination: Dict[str, Any], source: Dict[str, Any]) -> None:

    """Recursively merge ``source`` into ``destination`` in-place."""
    for key, value in source.items():
        if (
            isinstance(value, dict)
            and isinstance(destination.get(key), dict)
        ):
            _deep_update(destination[key], value)
        else:
            destination[key] = value


def _candidate_paths(path: Optional[str]) -> Iterable[Path]:
    """Yield YAML paths to probe for overrides."""
    if path:
        yield Path(path)
        return

    env_override = os.environ.get("VIZ_DEFAULTS_YAML")
    if env_override:
        yield Path(env_override)

    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parent

    yield project_root / "viz_defaults.yaml"
    yield Path.cwd() / "viz_defaults.yaml"
    yield package_dir / "viz_defaults.yaml"


def reload_viz_defaults(path: Optional[str] = None) -> None:
    """Reload visualization settings from YAML, falling back to built-ins."""
    defaults = copy.deepcopy(_DEFAULTS)

    if yaml:
        for candidate in _candidate_paths(path):
            if candidate.is_file():
                try:
                    data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
                except Exception:
                    continue
                if isinstance(data, dict):
                    _deep_update(defaults, data)
                break

    global _EFFECTIVE_DEFAULTS
    _EFFECTIVE_DEFAULTS = defaults


def get_viz_defaults() -> Dict[str, Any]:
    """Return a deep copy of the effective visualization defaults."""
    return copy.deepcopy(_EFFECTIVE_DEFAULTS)


def get_viz_section(section: str) -> Dict[str, Any]:
    """Return merged defaults for ``section`` including global fallbacks."""
    defaults = get_viz_defaults()
    merged: Dict[str, Any] = {}

    global_defaults = defaults.get("global", {})
    if isinstance(global_defaults, dict):
        merged = copy.deepcopy(global_defaults)

    section_defaults = defaults.get(section, {})
    if isinstance(section_defaults, dict):
        _deep_update(merged, copy.deepcopy(section_defaults))

    return merged


# Initialise configuration once on import.
reload_viz_defaults()
