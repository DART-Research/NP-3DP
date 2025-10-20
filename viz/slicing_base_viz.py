# -*- coding: utf-8 -*-
"""Interactive Plotly visualisations for :class:`~slicing.slicing_base.SlicingBaseGraph` outputs.

The helpers render scalar fields, boundary-aware distance maps, and graph connectivity
without mutating the underlying data structures.

Maintainer: Abdallah Kamhawi (PhD researcher, DART Laboratory; Kamhawi@umich.edu)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any, List, Dict
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
from loaders.mesh_loader import MeshLoader
from viz.plot_utils import guard_connection_reset

if TYPE_CHECKING:
    # Guarded import for type checking; avoids runtime cycles.
    from slicing.slicing_base import SlicingBaseGraph


from viz.config import get_viz_defaults, get_viz_section, reload_viz_defaults

_SECTION_NAME = "slicing_base"


def get_current_viz_defaults() -> Dict[str, Any]:
    """Return a deep copy of all visualization defaults."""
    return get_viz_defaults()


def _defaults() -> Dict[str, Any]:
    """Return the merged visual defaults for this module."""
    return get_viz_section(_SECTION_NAME)


# --------------------------------------------------------------------- #
# Colors / styles                                                       #
# --------------------------------------------------------------------- #
_LOWER_RGB = 'rgb(60,90,220)'    # bluish
_UPPER_RGB = 'rgb(220,70,60)'    # reddish
_INTERNAL_RGB = 'rgb(140,140,140)'


# ===================================================================== #
# 1) Visualize any per-vertex scalar on the *mesh*                      #
# ===================================================================== #
def visualize_scalar_field(
    mesh: MeshLoader,
    scalar_field: np.ndarray,
    *,
    colormap: Optional[str] = None,
    opacity: Optional[float] = None,
    title: str = "Scalar Field on Mesh",
    normalize: Optional[bool] = None,
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
    save_html: Optional[bool] = None,
    auto_open: Optional[bool] = None,
    filepath: Optional[str] = None,
) -> go.Figure:
    """Render a per-vertex scalar field on the triangular surface mesh.

    Parameters
    ----------
    mesh : MeshLoader
        Source mesh to render.
    scalar_field : np.ndarray
        Scalar values defined per vertex.
    colormap, opacity, title, normalize, save_html, auto_open, filepath : optional
        Visual configuration options overriding the defaults from :mod:`viz.config`.

    Returns
    -------
    go.Figure
        Plotly figure instance representing the coloured mesh.
    """
    if scalar_field.shape[0] != mesh.vertices.shape[0]:
        raise ValueError("scalar_field length must match number of mesh vertices")

    defaults = _defaults()
    filepaths = defaults.get("filepaths", {})

    colormap = defaults.get("colormap", "Turbo") if colormap is None else colormap
    opacity = defaults.get("opacity", 1.0) if opacity is None else opacity
    normalize = defaults.get("normalize", False) if normalize is None else normalize
    save_html = defaults.get("save_html", False) if save_html is None else save_html
    auto_open = defaults.get("auto_open", True) if auto_open is None else auto_open
    filepath = filepaths.get("scalar", "scalar_field.html") if filepath is None else filepath

    hide_axes = defaults.get("hide_axes", True)
    show_grid = bool(defaults.get("show_grid", False))
    aspectmode = defaults.get("aspectmode", "data")
    camera = defaults.get("camera")
    paper_bgcolor = defaults.get("paper_bgcolor", "white")
    plot_bgcolor = defaults.get("plot_bgcolor", "white")

    values = scalar_field.astype(float)
    if normalize:
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin < 1e-12:
            values = np.zeros_like(values)
        else:
            values = (values - vmin) / (vmax - vmin)

    faces = mesh.faces.astype(int)
    verts = mesh.vertices.astype(float)

    tri_kwargs: Dict[str, Any] = dict(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=values,
        colorscale=colormap,
        opacity=opacity,
        showscale=True,
        colorbar=dict(title=title),
    )
    if cmin is not None:
        tri_kwargs["cmin"] = float(cmin)
    if cmax is not None:
        tri_kwargs["cmax"] = float(cmax)
    tri = go.Mesh3d(**tri_kwargs)

    fig = go.Figure(data=[tri])

    scene_config: Dict[str, Any] = dict(aspectmode=aspectmode)
    if camera is not None:
        scene_config["camera"] = camera

    axis_cfg = dict(
        showgrid=show_grid,
        showbackground=False,
        showline=False,
        showspikes=False,
    )

    if hide_axes:
        axis_cfg.update(visible=False, showticklabels=False)
    else:
        axis_cfg.setdefault("visible", True)
        axis_cfg.setdefault("showticklabels", True)

    scene_config["xaxis"] = axis_cfg.copy()
    scene_config["yaxis"] = axis_cfg.copy()
    scene_config["zaxis"] = axis_cfg.copy()

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        scene=scene_config,
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    _handle_save_or_show(fig, save_html=save_html, auto_open=auto_open, filepath=filepath)
    return fig


# ===================================================================== #
# 2) Visualize the *graph* with boundaries and saddles                  #
# ===================================================================== #
def visualize_slicing_graph(
    sbg: "SlicingBaseGraph",
    *,
    scalar_field: Optional[np.ndarray] = None,
    title: str = "Slicing graph (boundaries & saddles)",
    colormap: Optional[str] = None,
    node_size: Optional[int] = None,
    show_internal_edges: Optional[bool] = None,
    internal_edge_alpha: Optional[float] = None,
    save_html: Optional[bool] = None,
    auto_open: Optional[bool] = None,
    filepath: Optional[str] = None,
) -> go.Figure:
    """Interactive visualization of the slicing graph with boundary context.

    Parameters
    ----------
    sbg : SlicingBaseGraph
        Graph instance to render.
    scalar_field : Optional[np.ndarray]
        Optional scalar field used to colour nodes. When ``None`` the field stored on
        the graph is used instead.
    title, colormap, node_size, show_internal_edges, internal_edge_alpha, save_html, auto_open, filepath : optional
        Visual configuration overrides.

    Returns
    -------
    go.Figure
        Plotly figure describing the slicing graph.
    """
    _require_boundary_sides(sbg)
    _require_saddle_annotations(sbg)

    defaults = _defaults()
    filepaths = defaults.get("filepaths", {})

    colormap = defaults.get("colormap", "Turbo") if colormap is None else colormap
    node_size = int(defaults.get("node_size", 4) if node_size is None else node_size)
    show_internal_edges = (
        defaults.get("show_internal_edges", True)
        if show_internal_edges is None
        else show_internal_edges
    )
    internal_edge_alpha = (
        float(defaults.get("internal_edge_alpha", 0.25))
        if internal_edge_alpha is None
        else float(internal_edge_alpha)
    )
    save_html = defaults.get("save_html", False) if save_html is None else save_html
    auto_open = defaults.get("auto_open", True) if auto_open is None else auto_open
    filepath = filepaths.get("graph", "slicing_graph.html") if filepath is None else filepath

    hide_axes = defaults.get("hide_axes", True)
    show_grid = bool(defaults.get("show_grid", False))
    aspectmode = defaults.get("aspectmode", "data")
    camera = defaults.get("camera")
    paper_bgcolor = defaults.get("paper_bgcolor", "white")
    plot_bgcolor = defaults.get("plot_bgcolor", "white")

    G = sbg.G
    node_ids = sorted(G.nodes())
    pos = np.array([G.nodes[n]['pos'] for n in node_ids], dtype=float)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

    def _edge_segments(filter_fn):
        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        for u, v in G.edges():
            if not filter_fn(u, v):
                continue
            p0, p1 = G.nodes[u]['pos'], G.nodes[v]['pos']
            xs += [p0[0], p1[0], None]
            ys += [p0[1], p1[1], None]
            zs += [p0[2], p1[2], None]
        return xs, ys, zs

    elx, ely, elz = _edge_segments(lambda u, v: G[u][v].get('edge_type') == 'boundary'
                                               and G[u][v].get('boundary_side') == 'lower')
    eux, euy, euz = _edge_segments(lambda u, v: G[u][v].get('edge_type') == 'boundary'
                                               and G[u][v].get('boundary_side') == 'upper')
    if show_internal_edges:
        eix, eiy, eiz = _edge_segments(lambda u, v: G[u][v].get('edge_type') != 'boundary')
    else:
        eix, eiy, eiz = [], [], []

    tr_lower = go.Scatter3d(
        x=elx,
        y=ely,
        z=elz,
        mode='lines',
        line=dict(width=2, color=_LOWER_RGB),
        name='boundary: lower',
        opacity=0.95,
    )
    tr_upper = go.Scatter3d(
        x=eux,
        y=euy,
        z=euz,
        mode='lines',
        line=dict(width=2, color=_UPPER_RGB),
        name='boundary: upper',
        opacity=0.95,
    )
    tr_internal = go.Scatter3d(
        x=eix,
        y=eiy,
        z=eiz,
        mode='lines',
        line=dict(width=1, color=_INTERNAL_RGB),
        name='internal edges',
        opacity=float(internal_edge_alpha),
    )

    if scalar_field is not None:
        vals = np.asarray(scalar_field, dtype=float)
        if vals.shape[0] != len(node_ids):
            raise ValueError("scalar_field length must match number of graph nodes")
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        rng = vmax - vmin if vmax > vmin else 1.0
        cvals = (vals - vmin) / rng
        node_trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=node_size,
                color=cvals,
                colorscale=colormap,
                opacity=0.9,
                colorbar=dict(title="Scalar (norm)", thickness=18, len=0.55, x=1.02),
            ),
            name='nodes',
        )
    else:
        side = np.array([G.nodes[n].get('boundary_side', 'intra') for n in node_ids])
        side_map = {'lower': 0.0, 'intra': 0.5, 'upper': 1.0}
        cvals = np.vectorize(lambda s: side_map.get(s, 0.5))(side)
        node_trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=node_size,
                color=cvals,
                colorscale='RdBu',
                opacity=0.9,
                colorbar=dict(title="Side", thickness=18, len=0.55, x=1.02),
            ),
            name='nodes',
        )

    saddle_nodes = [n for n in node_ids if G.nodes[n].get('morse_index') == 1 or G.nodes[n].get('critical_type') == 'saddle']
    sp = np.array([G.nodes[n]['pos'] for n in saddle_nodes], dtype=float)
    tr_saddle = go.Scatter3d(
        x=sp[:, 0],
        y=sp[:, 1],
        z=sp[:, 2],
        mode='markers',
        marker=dict(size=max(6, node_size + 2), symbol='x', color='red', line=dict(width=1)),
        name='saddles',
    )

    traces: List[Any] = [node_trace, tr_lower, tr_upper]
    if show_internal_edges:
        traces.append(tr_internal)
    traces.append(tr_saddle)

    axis_cfg = dict(
        showgrid=show_grid,
        showbackground=False,
        showline=False,
        showspikes=False,
        title='',
    )
    if hide_axes:
        axis_cfg.update(visible=False, showticklabels=False)
    else:
        axis_cfg.setdefault("visible", True)
        axis_cfg.setdefault("showticklabels", True)

    scene_dict: Dict[str, Any] = dict(aspectmode=aspectmode)
    if camera is not None:
        scene_dict["camera"] = camera
    scene_dict["xaxis"] = axis_cfg.copy()
    scene_dict["yaxis"] = axis_cfg.copy()
    scene_dict["zaxis"] = axis_cfg.copy()

    layout = go.Layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=scene_dict,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
    )

    fig = go.Figure(data=traces, layout=layout)
    _handle_save_or_show(fig, save_html=save_html, auto_open=auto_open, filepath=filepath)
    return fig


# ===================================================================== #
# 3) Subset scalar fields: UPPER / LOWER / SADDLES                      #
# ===================================================================== #
def visualize_scalar_field_upper_only(
    sbg: "SlicingBaseGraph",
    *,
    colormap: Optional[str] = None,
    title: str = 'Min distance to UPPER boundaries',
    normalize: Optional[bool] = None,
    save_html: Optional[bool] = None,
    auto_open: Optional[bool] = None,
    filepath: Optional[str] = None,
) -> go.Figure:
    """Visualise the minimum geodesic distance to the upper boundary loops.

    Parameters
    ----------
    sbg : SlicingBaseGraph
        Graph instance providing geodesic distances and boundary labels.
    colormap, title, normalize, save_html, auto_open, filepath : optional
        Visual configuration overrides.

    Returns
    -------
    go.Figure
        Plotly figure visualising the distance field.
    """
    defaults = _defaults()
    filepaths = defaults.get("filepaths", {})

    colormap = defaults.get("colormap", "Turbo") if colormap is None else colormap
    normalize = defaults.get("normalize", False) if normalize is None else normalize
    save_html = defaults.get("save_html", False) if save_html is None else save_html
    auto_open = defaults.get("auto_open", True) if auto_open is None else auto_open
    filepath = filepaths.get("upper", "scalar_upper.html") if filepath is None else filepath

    field = _field_min_distance_to_boundary_side(sbg, 'upper')
    fig = visualize_scalar_field(
        sbg.mesh,
        field,
        colormap=colormap,
        title=title,
        normalize=normalize,
        save_html=False,
        auto_open=False,
    )
    _handle_save_or_show(fig, save_html=save_html, auto_open=auto_open, filepath=filepath)
    return fig


def visualize_scalar_field_lower_only(
    sbg: "SlicingBaseGraph",
    *,
    colormap: Optional[str] = None,
    title: str = 'Min distance to LOWER boundaries',
    normalize: Optional[bool] = None,
    save_html: Optional[bool] = None,
    auto_open: Optional[bool] = None,
    filepath: Optional[str] = None,
) -> go.Figure:
    """Visualise the minimum geodesic distance to the lower boundary loops.

    Parameters
    ----------
    sbg : SlicingBaseGraph
        Graph instance providing geodesic distances and boundary labels.
    colormap, title, normalize, save_html, auto_open, filepath : optional
        Visual configuration overrides.

    Returns
    -------
    go.Figure
        Plotly figure visualising the distance field.
    """
    defaults = _defaults()
    filepaths = defaults.get("filepaths", {})

    colormap = defaults.get("colormap", "Turbo") if colormap is None else colormap
    normalize = defaults.get("normalize", False) if normalize is None else normalize
    save_html = defaults.get("save_html", False) if save_html is None else save_html
    auto_open = defaults.get("auto_open", True) if auto_open is None else auto_open
    filepath = filepaths.get("lower", "scalar_lower.html") if filepath is None else filepath

    field = _field_min_distance_to_boundary_side(sbg, 'lower')
    fig = visualize_scalar_field(
        sbg.mesh,
        field,
        colormap=colormap,
        title=title,
        normalize=normalize,
        save_html=False,
        auto_open=False,
    )
    _handle_save_or_show(fig, save_html=save_html, auto_open=auto_open, filepath=filepath)
    return fig


def visualize_scalar_field_saddles_only(
    sbg: "SlicingBaseGraph",
    *,
    colormap: Optional[str] = None,
    title: str = 'Min distance to saddle vertices',
    normalize: Optional[bool] = None,
    save_html: Optional[bool] = None,
    auto_open: Optional[bool] = None,
    filepath: Optional[str] = None,
) -> go.Figure:
    """Visualise the minimum geodesic distance to annotated saddle vertices.

    Parameters
    ----------
    sbg : SlicingBaseGraph
        Graph instance with saddle annotations.
    colormap, title, normalize, save_html, auto_open, filepath : optional
        Visual configuration overrides.

    Returns
    -------
    go.Figure
        Plotly figure visualising the distance field.
    """
    defaults = _defaults()
    filepaths = defaults.get("filepaths", {})

    colormap = defaults.get("colormap", "Turbo") if colormap is None else colormap
    normalize = defaults.get("normalize", False) if normalize is None else normalize
    save_html = defaults.get("save_html", False) if save_html is None else save_html
    auto_open = defaults.get("auto_open", True) if auto_open is None else auto_open
    filepath = filepaths.get("saddles", "scalar_saddles.html") if filepath is None else filepath

    field = _field_min_distance_to_saddles(sbg)
    fig = visualize_scalar_field(
        sbg.mesh,
        field,
        colormap=colormap,
        title=title,
        normalize=normalize,
        save_html=False,
        auto_open=False,
    )
    _handle_save_or_show(fig, save_html=save_html, auto_open=auto_open, filepath=filepath)
    return fig


# ===================================================================== #
# Internal helpers                                                      #
# ===================================================================== #
def _handle_save_or_show(
    fig: go.Figure,
    *,
    save_html: bool,
    auto_open: bool,
    filepath: str,
) -> None:
    """Persist or display ``fig`` according to the configured I/O policy."""
    if save_html:
        guard_connection_reset(
            lambda: plot(fig, filename=filepath, auto_open=auto_open, include_plotlyjs=True),
            context=f"saving Plotly HTML to {filepath}",
        )
    elif auto_open:
        guard_connection_reset(fig.show, context="opening Plotly viewer")



def _require_boundary_sides(sbg: "SlicingBaseGraph") -> None:
    """
    Ensure upper/lower classification exists on the graph.

    Raises
    ------
    RuntimeError
        If no boundary sides are present. Call:
            sbg.label_lower_upper_boundaries(...)
        before visualization.
    """
    if not hasattr(sbg, 'boundary_sides') or not sbg.boundary_sides:
        raise RuntimeError(
            "Boundary sides are missing. Please call "
            "`SlicingBaseGraph.label_lower_upper_boundaries(...)` first."
        )


def _require_saddle_annotations(sbg: "SlicingBaseGraph") -> None:
    """
    Ensure at least one node is marked as a saddle.

    Raises
    ------
    RuntimeError
        If no node carries saddle labels (`morse_index==1` or `critical_type=='saddle'`).
    """
    for _, d in sbg.G.nodes(data=True):
        mi = d.get('morse_index', None)
        ct = d.get('critical_type', None)
        if mi == 1 or ct == 'saddle':
            return
    raise RuntimeError(
        "Saddle annotations are missing. Annotate saddles on nodes first "
        "(e.g., `sbg.detect_saddle_points(..., annotate=True)` in your workflow) "
        "before using this visualization."
    )


def _field_min_distance_to_boundary_side(
    sbg: "SlicingBaseGraph",
    side: str,
) -> np.ndarray:
    """
    Build a per-vertex field equal to the **minimum geodesic distance**
    to the set of boundary vertices on the specified side.

    Parameters
    ----------
    side : {'upper','lower'}

    Returns
    -------
    field : (V,) float
        Minimum geodesic distances.

    Raises
    ------
    RuntimeError
        If boundary sides were not labeled or if no loop exists on that side.
    """
    side = side.lower()
    if side not in ('upper', 'lower'):
        raise ValueError("side must be 'upper' or 'lower'")

    _require_boundary_sides(sbg)

    loops = sbg._boundary_loops_data()
    if not loops:
        raise RuntimeError("No boundary loops found on the mesh.")

    side_bnums = [b for b, s in getattr(sbg, 'boundary_sides', {}).items() if s == side]
    if not side_bnums:
        raise RuntimeError(f"No boundary loops classified as '{side}'.")

    dists: List[np.ndarray] = [sbg.compute_geodesic_distance_from_boundary(int(b)) for b in side_bnums]
    D = np.vstack(dists).T  # (V, number_of_side_loops)
    return D.min(axis=1)


def _field_min_distance_to_saddles(sbg: "SlicingBaseGraph") -> np.ndarray:
    """
    Build a per-vertex field equal to the **minimum geodesic distance**
    to the set of saddle vertices.

    Returns
    -------
    field : (V,) float
        Minimum geodesic distances to the saddle set.

    Raises
    ------
    RuntimeError
        If no saddle annotations exist on the graph.
    """
    annotated = [n for n, d in sbg.G.nodes(data=True)
                 if d.get('morse_index') == 1 or d.get('critical_type') == 'saddle']
    if not annotated:
        raise RuntimeError(
            "No annotated saddle vertices found on the graph. Annotate first "
            "before building the saddle-distance field."
        )
    saddles = np.array(annotated, dtype=int)
    return sbg.compute_geodesic_distance_to_nodes(saddles.tolist())
