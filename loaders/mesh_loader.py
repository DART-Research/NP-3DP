"""Mesh loading helpers with optional scalar-field visualisation.

Maintainer: Abdallah Kamhawi (PhD researcher, DART Laboratory; Kamhawi@umich.edu)
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objs as go
import trimesh
from matplotlib import cm
from plotly.offline import plot


class MeshLoader(trimesh.Trimesh):
    """Extend :class:`trimesh.Trimesh` with colour-aware utilities."""

    def __init__(self, file_path: str, **kwargs):
        """Load a surface mesh and retain all original visual attributes.

        Parameters
        ----------
        file_path : str
            Path to the mesh file. All formats supported by
            :func:`trimesh.load_mesh` are accepted.
        **kwargs : dict, optional
            Additional keyword arguments forwarded to
            :func:`trimesh.load_mesh`.

        Raises
        ------
        TypeError
            If the loaded geometry is not a single
            :class:`trimesh.Trimesh` instance.
        """
        mesh = trimesh.load_mesh(file_path, **kwargs)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Loaded geometry is not a single trimesh.Trimesh")
        super().__init__(
            vertices=mesh.vertices,
            faces=mesh.faces,
            process=False,
            metadata=mesh.metadata,
        )
        # Preserve textures/materials so downstream exports mirror the source asset.
        self.visual = mesh.visual

    def color_by_vertex_attribute(self, values: np.ndarray) -> trimesh.Trimesh:
        """Return a mesh copy coloured by a per-vertex scalar attribute.

        The routine linearly normalises ``values`` into ``[0, 1]`` via
        ``(values - v_min) / (v_max - v_min)`` before mapping them through
        Matplotlib's ``Turbo`` colormap. Degenerate inputs with
        ``v_min == v_max`` collapse to zeros which yields a uniform colour.

        Parameters
        ----------
        values : np.ndarray
            Array of length ``n_vertices`` containing the scalar field to
            visualise.

        Returns
        -------
        trimesh.Trimesh
            Copy of ``self`` with ``vertex_colors`` encoded as RGBA ``uint8``.

        Raises
        ------
        ValueError
            If the length of ``values`` does not match the vertex count.
        """
        n_verts = self.vertices.shape[0]
        if values.shape[0] != n_verts:
            raise ValueError(
                f"values length ({values.shape[0]}) != vertex count ({n_verts})"
            )

        vmin = float(values.min())
        vmax = float(values.max())
        if np.isclose(vmin, vmax):
            normed = np.zeros_like(values, dtype=float)
        else:
            normed = (values - vmin) / (vmax - vmin)

        cmap = cm.get_cmap("turbo")
        rgba = cmap(np.clip(normed, 0.0, 1.0))
        vertex_colors = (rgba * 255).astype(np.uint8)

        coloured_mesh = self.copy()
        coloured_mesh.visual.vertex_colors = vertex_colors
        return coloured_mesh

    def visualize(
        self,
        scalar_field: np.ndarray | None = None,
        filepath: str = "mesh_plot.html",
    ) -> None:
        """Render the mesh with Plotly and export an interactive HTML view.

        Parameters
        ----------
        scalar_field : np.ndarray, optional
            Optional per-vertex scalar field. When provided the colours are
            obtained from the ``Turbo`` colormap using the same linear rule
            as :meth:`color_by_vertex_attribute`.
        filepath : str, optional
            Destination HTML path written by :func:`plotly.offline.plot`.

        Raises
        ------
        ValueError
            If ``scalar_field`` does not align with the vertex count.
        """
        verts = self.vertices
        faces = self.faces

        mesh_kwargs = dict(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=1.0,
        )

        if scalar_field is not None:
            if scalar_field.shape[0] != verts.shape[0]:
                raise ValueError(
                    f"scalar_field length ({scalar_field.shape[0]}) "
                    f"!= vertex count ({verts.shape[0]})"
                )
            trace = go.Mesh3d(
                **mesh_kwargs,
                intensity=scalar_field,
                colorscale="Turbo",
                showscale=True,
            )
        else:
            trace = go.Mesh3d(**mesh_kwargs, color="lightgray")

        layout = go.Layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
            ),
            margin=dict(t=0, l=0, r=0, b=0),
        )

        fig = go.Figure(data=[trace], layout=layout)
        plot(fig, filename=filepath)



