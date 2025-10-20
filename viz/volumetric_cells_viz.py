"""Visualization helpers for volumetric slice cells."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot

from analysis.volumetric_cells import VolumetricAnalysisResult, VolumetricCell
from viz.plot_utils import guard_connection_reset
from loaders.mesh_loader import MeshLoader
from viz.config import get_viz_section
from viz.slicing_base_viz import visualize_scalar_field

_SECTION = "volumetric_cells"

_TRIANGLE_TEMPLATE: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (0, 2, 3),  # start face
    (4, 6, 5), (4, 7, 6),  # end face
    (0, 4, 7), (0, 7, 3),  # sides
    (1, 5, 6), (1, 6, 2),
    (0, 1, 5), (0, 5, 4),
    (3, 2, 6), (3, 6, 7),
)


def visualize_volumetric_cells(
    result: VolumetricAnalysisResult,
    *,
    metric: str = "cusp_height",
    slice_indices: Optional[Sequence[int]] = None,
    colorscale: Optional[str] = None,
    opacity: Optional[float] = None,
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
    save_html: Optional[bool] = None,
    auto_open: Optional[bool] = None,
    filepath: Optional[str] = None,
    title: Optional[str] = None,
    showscale: Optional[bool] = None,
) -> go.Figure:
    """Render volumetric cells coloured by the requested metric."""
    defaults = get_viz_section(_SECTION)
    filepaths = defaults.get("filepaths", {})

    colorscale = colorscale or defaults.get("colorscale", "Turbo")
    opacity = defaults.get("opacity", 0.6) if opacity is None else opacity
    save_html = defaults.get("save_html", False) if save_html is None else save_html
    auto_open = defaults.get("auto_open", True) if auto_open is None else auto_open
    filepath = filepaths.get("cells", "volumetric_cells.html") if filepath is None else filepath
    title = title or defaults.get("title", "Volumetric Cells")

    cells = list(result.iter_cells())
    if slice_indices is not None:
        mask = set(slice_indices)
        cells = [cell for cell in cells if cell.slice_index in mask]
    if not cells:
        raise ValueError("No cells available for visualization")

    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    tri_i: List[int] = []
    tri_j: List[int] = []
    tri_k: List[int] = []
    tri_values: List[float] = []

    per_cell_values: List[float] = []

    vertex_offset = 0
    for cell in cells:
        vertices = cell.vertices
        if vertices.shape[0] != 8:
            raise ValueError("Each volumetric cell must contain 8 vertices for visualization")
        xs.extend(vertices[:, 0].tolist())
        ys.extend(vertices[:, 1].tolist())
        zs.extend(vertices[:, 2].tolist())

        metric_value = cell.metric(metric)
        if not np.isfinite(metric_value):
            metric_value = float("nan")
        per_cell_values.append(metric_value)

        for tri in _TRIANGLE_TEMPLATE:
            a, b, c = tri
            tri_i.append(vertex_offset + a)
            tri_j.append(vertex_offset + b)
            tri_k.append(vertex_offset + c)
            tri_values.append(metric_value if np.isfinite(metric_value) else 0.0)
        vertex_offset += vertices.shape[0]

    finite_values = [val for val in per_cell_values if np.isfinite(val)]
    showscale = showscale if showscale is not None else bool(finite_values)
    # Allow caller to impose cmin/cmax (e.g., for cusp height against layer height)
    cmin_val = cmin if cmin is not None else (min(finite_values) if finite_values else 0.0)
    cmax_val = cmax if cmax is not None else (max(finite_values) if finite_values else 1.0)
    if not finite_values and (cmin is None or cmax is None):
        tri_values = [0.0 for _ in tri_values]

    mesh = go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=tri_i,
        j=tri_j,
        k=tri_k,
        opacity=opacity,
        intensity=tri_values,
        intensitymode="cell",
        colorscale=colorscale,
        cmin=cmin_val,
        cmax=cmax_val,
        showscale=showscale,
        flatshading=True,
    )

    fig = go.Figure(data=[mesh])
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(t=40, l=0, r=0, b=0),
        showlegend=False,
    )

    if save_html:
        guard_connection_reset(
            lambda: plot(fig, filename=filepath, auto_open=auto_open, include_plotlyjs=True),
            context=f"saving volumetric cells figure to {filepath}",
        )
    elif auto_open:
        guard_connection_reset(fig.show, context="opening volumetric cells figure")
    return fig


def visualize_volumetric_structure(
    result: VolumetricAnalysisResult,
    *,
    slice_indices: Optional[Sequence[int]] = None,
    color: str = "#B0B0B0",
    opacity: float = 0.35,
    save_html: Optional[bool] = None,
    auto_open: Optional[bool] = None,
    filepath: Optional[str] = None,
    title: str = "Volumetric Discretization",
) -> go.Figure:
    """Render the volumetric discretisation with a uniform colour."""
    defaults = get_viz_section(_SECTION)
    filepaths = defaults.get("filepaths", {})

    save_html = defaults.get("save_html", False) if save_html is None else save_html
    auto_open = defaults.get("auto_open", True) if auto_open is None else auto_open
    filepath = filepath or filepaths.get("structure", "volumetric_cells_structure.html")

    cells = list(result.iter_cells())
    if slice_indices is not None:
        mask = set(slice_indices)
        cells = [cell for cell in cells if cell.slice_index in mask]
    if not cells:
        raise ValueError("No cells available for visualization")

    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    tri_i: List[int] = []
    tri_j: List[int] = []
    tri_k: List[int] = []

    vertex_offset = 0
    for cell in cells:
        vertices = cell.vertices
        xs.extend(vertices[:, 0].tolist())
        ys.extend(vertices[:, 1].tolist())
        zs.extend(vertices[:, 2].tolist())
        for tri in _TRIANGLE_TEMPLATE:
            a, b, c = tri
            tri_i.append(vertex_offset + a)
            tri_j.append(vertex_offset + b)
            tri_k.append(vertex_offset + c)
        vertex_offset += vertices.shape[0]

    mesh = go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=tri_i,
        j=tri_j,
        k=tri_k,
        opacity=opacity,
        color=color,
        flatshading=True,
        showscale=False,
    )

    fig = go.Figure(data=[mesh])
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(t=40, l=0, r=0, b=0),
        showlegend=False,
    )

    if save_html:
        guard_connection_reset(
            lambda: plot(fig, filename=filepath, auto_open=auto_open, include_plotlyjs=True),
            context=f"saving volumetric structure figure to {filepath}",
        )
    elif auto_open:
        guard_connection_reset(fig.show, context="opening volumetric structure figure")
    return fig


def visualize_volumetric_metric_on_mesh(
    result: VolumetricAnalysisResult,
    mesh: MeshLoader,
    *,
    metric: str = "cusp_height",
    method: Optional[str] = None,
    colorscale: Optional[str] = None,
    opacity: Optional[float] = None,
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
    save_html: Optional[bool] = None,
    auto_open: Optional[bool] = None,
    filepath: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    defaults = get_viz_section(_SECTION)
    filepaths = defaults.get("filepaths", {})
    mesh_defaults = defaults.get("mesh", {})

    colorscale = colorscale or mesh_defaults.get("colorscale") or defaults.get("colorscale", "Turbo")
    opacity = opacity if opacity is not None else mesh_defaults.get("opacity", 1.0)
    save_html = save_html if save_html is not None else defaults.get("save_html", False)
    auto_open = auto_open if auto_open is not None else defaults.get("auto_open", True)
    filepath = filepath or filepaths.get("mesh", "volumetric_cells_mesh.html")
    title = title or mesh_defaults.get("title", "Volumetric Metric on Mesh")
    method = method or mesh_defaults.get("method", "nearest")

    field = result.map_metric_to_mesh(mesh, metric=metric, method=method)
    if cmin is not None or cmax is not None:
        if cmin is not None:
            field = np.maximum(field, float(cmin))
        if cmax is not None:
            field = np.minimum(field, float(cmax))
    fig = visualize_scalar_field(
        mesh,
        field,
        colormap=colorscale,
        opacity=opacity,
        title=title,
        cmin=cmin,
        cmax=cmax,
        save_html=save_html,
        auto_open=auto_open,
        filepath=filepath,
    )
    return fig


__all__ = [
    "visualize_volumetric_cells",
    "visualize_volumetric_structure",
    "visualize_volumetric_metric_on_mesh",
]
