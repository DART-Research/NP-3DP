"""JSON export for per-vertex scalar fields."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def export_scalar_field_to_json(
    mesh: Any,
    scalar_field: Sequence[float] | np.ndarray,
    output_dir: str | Path,
    *,
    filename: str = "scalar_field.json",
    indent: int | None = 2,
    include_faces: bool = True,
) -> Path:
    """Serialise a per-vertex scalar field to JSON.

    Parameters
    ----------
    mesh
        Mesh instance exposing ``vertices`` (and optionally ``faces``) arrays.
        Accepts :class:`loaders.mesh_loader.MeshLoader`, :class:`trimesh.Trimesh`,
        or any compatible object implementing the attributes.
    scalar_field
        Iterable of numeric values aligned with ``mesh.vertices``.
    output_dir
        Destination directory for the JSON export.
    filename
        Name of the JSON file to create inside ``output_dir``.
    indent
        Indentation passed through to :func:`json.dumps`. Set to ``None`` for a compact export.
    include_faces
        When ``True`` and the mesh defines ``faces``, face connectivity is written alongside vertices.

    Returns
    -------
    Path
        Absolute path to the written JSON file.

    Raises
    ------
    TypeError
        If ``mesh`` does not define a ``vertices`` attribute.
    ValueError
        When ``scalar_field`` does not align with the vertex count or contains non-finite values.
    """
    if mesh is None or not hasattr(mesh, "vertices"):
        raise TypeError("mesh must provide a vertices attribute shaped (V, 3)")

    vertices = np.asarray(getattr(mesh, "vertices"), dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("mesh.vertices must be shaped (V, 3)")

    values = np.asarray(scalar_field, dtype=float).reshape(-1)
    vertex_count = vertices.shape[0]
    if values.size != vertex_count:
        raise ValueError(
            f"scalar_field length ({values.size}) does not match vertex count ({vertex_count})"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("scalar_field contains non-finite entries")

    faces_array = None
    face_count = 0
    if include_faces and hasattr(mesh, "faces"):
        faces = np.asarray(getattr(mesh, "faces"))
        if faces.size > 0:
            if faces.ndim != 2 or faces.shape[1] not in (3, 4):
                raise ValueError("Only triangular or quad face connectivity is supported")
            faces_array = faces.astype(int, copy=False)
            face_count = faces_array.shape[0]

    stats = {
        "min": float(values.min(initial=float("inf"))),
        "max": float(values.max(initial=float("-inf"))),
        "mean": float(values.mean() if values.size else 0.0),
        "std": float(values.std(ddof=0) if values.size else 0.0),
    }
    stats["range"] = float(stats["max"] - stats["min"])

    bounding_box = {
        "min": vertices.min(axis=0).astype(float).tolist(),
        "max": vertices.max(axis=0).astype(float).tolist(),
    }

    payload: dict[str, Any] = {
        "meta": {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "vertex_count": vertex_count,
            "face_count": face_count,
            "field_statistics": stats,
            "bounding_box": bounding_box,
        },
        "vertices": vertices.astype(float).tolist(),
        "scalar_field": values.astype(float).tolist(),
    }

    if faces_array is not None:
        payload["faces"] = faces_array.tolist()

    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename
    target.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    return target.resolve()
