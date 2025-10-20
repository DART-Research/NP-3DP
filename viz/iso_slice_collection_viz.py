"""Plotly-based helpers for visualising iso-slice collections."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
from collections import defaultdict
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot

from slicing.iso_slice_collection import IsoSliceCollection
from viz.plot_utils import guard_connection_reset
from viz.config import get_viz_section, reload_viz_defaults

_SECTION_NAME = "iso_slices"


def _defaults() -> Dict[str, Any]:

    """Return the merged visual defaults for this module."""
    return get_viz_section(_SECTION_NAME)


def _collect_component_curves(slice_obj) -> Sequence[np.ndarray]:
    """Return every curve to plot for a given slice.

    Parameters
    ----------
    slice_obj : IsoSlice
        Slice whose components should be visualised.

    Returns
    -------
    Sequence[np.ndarray]
        List of component polylines shaped ``(N_i, 3)``.
    """
    getter = getattr(slice_obj, "get_component_curves", None)
    if callable(getter):
        curves = getter()
        if curves:
            return [np.asarray(curve, dtype=float) for curve in curves]
    curve = getattr(slice_obj, "curve", None)
    if curve is None:
        return []
    return [np.asarray(curve, dtype=float)]


def _handle_output(fig: go.Figure, *, save_html: bool, auto_open: bool, filepath: str) -> None:
    """Persist or display ``fig`` according to the configured policy.

    Parameters
    ----------
    fig : go.Figure
        Figure to render.
    save_html : bool
        Write the figure to ``filepath`` when ``True``.
    auto_open : bool
        Open the HTML file or native viewer immediately.
    filepath : str
        Target path for the exported HTML file.
    """
    if save_html:
        guard_connection_reset(
            lambda: plot(fig, filename=filepath, auto_open=auto_open, include_plotlyjs=True),
            context=f"saving iso-slice figure to {filepath}",
        )
    elif auto_open:
        guard_connection_reset(fig.show, context="opening iso-slice figure")


def visualize_slices(
    collection: IsoSliceCollection,
    *,
    save_html: Optional[bool] = None,
    auto_open: Optional[bool] = None,
    filepath: Optional[str] = None,
    colorscale: Optional[Union[str, Sequence]] = None,
    line_width: Optional[float] = None,
    show_colorbar: Optional[bool] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Render every curve in ``collection`` using a Plotly colourscale.

    Parameters
    ----------
    collection : IsoSliceCollection
        Collection of iso-slices to plot.
    save_html, auto_open, filepath, colorscale, line_width, show_colorbar, title : optional
        Visual configuration overrides.

    Returns
    -------
    go.Figure
        Plotly figure containing all slice curves.
    """
    if len(collection) == 0:
        raise ValueError("IsoSliceCollection is empty; nothing to visualise")

    defaults = _defaults()
    filepaths = defaults.get("filepaths", {})

    total_slices = len(collection)
    level_tracker: defaultdict[float, int] = defaultdict(int)
    slice_meta = []
    for idx, slice_obj in enumerate(collection.slices):
        level_value = float(getattr(slice_obj, "level", 0.0))
        key = round(level_value, 6)
        occurrence = level_tracker[key]
        label = f"{idx + 1}.{occurrence}"
        level_tracker[key] += 1
        slice_meta.append((slice_obj, level_value, label))

    save_html = defaults.get("save_html", False) if save_html is None else save_html
    auto_open = defaults.get("auto_open", True) if auto_open is None else auto_open
    filepath = filepaths.get("slices", "slices.html") if filepath is None else filepath
    colorscale = defaults.get("colorscale", "Turbo") if colorscale is None else colorscale
    line_width = defaults.get("line_width", 2.5) if line_width is None else line_width
    show_colorbar = defaults.get("show_colorbar", True) if show_colorbar is None else show_colorbar

    hide_axes = defaults.get("hide_axes", True)
    show_grid = bool(defaults.get("show_grid", False))
    aspectmode = defaults.get("aspectmode", "data")
    camera = defaults.get("camera")
    paper_bgcolor = defaults.get("paper_bgcolor", "white")
    plot_bgcolor = defaults.get("plot_bgcolor", "white")

    default_title = defaults.get("title", "Iso-slices")
    if title is None:
        title = f"{default_title} (total={total_slices})"

    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    zs: List[Optional[float]] = []
    levels: List[float] = []
    customdata: List[List[Optional[object]]] = []

    for slice_obj, level_value, label in slice_meta:
        component_curves = _collect_component_curves(slice_obj)
        if not component_curves:
            continue
        for curve in component_curves:
            if curve.size == 0:
                continue
            for x, y, z in curve:
                xs.append(float(x))
                ys.append(float(y))
                zs.append(float(z))
                levels.append(level_value)
                customdata.append([level_value, label])
            xs.append(None)
            ys.append(None)
            zs.append(None)
            levels.append(np.nan)
            customdata.append([np.nan, None])

    finite_levels = [v for v in levels if np.isfinite(v)]
    if not finite_levels:
        raise ValueError("No valid curve geometry found in collection")

    cmin, cmax = min(finite_levels), max(finite_levels)

    hovertemplate = (
        f"Slice: %{{customdata[1]}} of {total_slices}<br>"
        "Level: %{customdata[0]:.4f}<br>"
        "X: %{x:.3f}<br>"
        "Y: %{y:.3f}<br>"
        "Z: %{z:.3f}<br>"
        "<extra></extra>"
    )

    trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(
            color=levels,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            width=line_width,
            colorbar=dict(title="Iso-level", thickness=20, len=0.7, x=1.02) if show_colorbar else None,
        ),
        customdata=customdata,
        hovertemplate=hovertemplate,
        showlegend=False,
    )

    fig = go.Figure(data=[trace])

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

    layout_config: Dict[str, Any] = dict(
        scene=scene_config,
        margin=dict(t=30 if title else 0, l=0, r=0, b=0),
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        title=dict(text=title, x=0.5, xanchor="center") if title else None,
    )

    fig.update_layout(**{k: v for k, v in layout_config.items() if v is not None})

    _handle_output(fig, save_html=save_html, auto_open=auto_open, filepath=filepath)
    return fig
