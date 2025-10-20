"""Graph-based slicing utilities for extracting topology-aware iso-curves.

The module builds a weighted vertex graph from a triangle mesh, decorates it with
boundary labels and geodesic metrics, and exposes helpers for scalar-field
construction and adaptive iso-slice extraction. The implementation keeps the
mathematical steps explicit so that the behaviour of the slicing controller can
be audited and reproduced.

Maintainer: Abdallah Kamhawi (PhD researcher, DART Laboratory; Kamhawi@umich.edu)
"""

from tqdm import tqdm
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Sequence, Optional, Any
from collections import defaultdict

from loaders.mesh_loader import MeshLoader
from slicing.iso_slice import IsoSlice
from slicing.multi_component_slicing import build_slice_components
from slicing.iso_slice_collection import IsoSliceCollection
from slicing.critical_points import classify_critical_points  # modular helper

from slicing.numba_accel import (
    HAS_NUMBA,
    compute_face_grad_norms_jit,
    dijkstra_multi_source_jit,
    dijkstra_single_source_jit,
    extract_iso_segments_jit,
)


class SlicingBaseGraph:
    """
    Graph representation of a 3D mesh suitable for geodesic distances and iso-slice extraction.
    """

    # --------------------------------------------------------------------- #
    #                           Construction & I/O                          #
    # --------------------------------------------------------------------- #
    def __init__(self, mesh: MeshLoader, *, show_progress: bool = True):
        """Build the graph representation of a mesh ready for geodesic slicing.
        
        Parameters
        ----------
        mesh : MeshLoader
            Triangular surface mesh to wrap.
        show_progress : bool, optional
            Enable progress bars for long-running routines."""
        if not isinstance(mesh, MeshLoader):
            raise TypeError("SlicingBaseGraph requires a MeshLoader instance")
        self.mesh = mesh
        self.show_progress = bool(show_progress)
        self.G = nx.Graph()
        self.border_count = 0
        self.border_geodesic_distances: Dict[int, np.ndarray] = {}
        self.custom_geodesic_distances: Dict[Tuple[int, ...], np.ndarray] = {}
        self._build_graph()
        self._label_boundaries()

        self._sorted_nodes: List[int] = sorted(self.G.nodes())
        self._node_index = {int(n): idx for idx, n in enumerate(self._sorted_nodes)}
        self._csr_indptr: Optional[np.ndarray] = None
        self._csr_indices: Optional[np.ndarray] = None
        self._csr_weights: Optional[np.ndarray] = None

    def _build_graph(self) -> None:
        """Populate the internal NetworkX graph with mesh vertices and Euclidean edge weights."""
        verts = self.mesh.vertices
        # Nodes
        for idx, (x, y, z) in tqdm(
            enumerate(verts),
            total=len(verts),
            disable=not self.show_progress,
            desc="Building graph nodes",
            unit="node",
        ):
            self.G.add_node(idx, pos=(float(x), float(y), float(z)))
        # Edges
        edges = self.mesh.edges_unique
        for v0, v1 in tqdm(
            edges,
            total=len(edges),
            disable=not self.show_progress,
            desc="Building graph edges",
            unit="edge",
        ):
            v0, v1 = int(v0), int(v1)
            p0, p1 = verts[v0], verts[v1]
            dist = float(np.linalg.norm(p0 - p1))
            self.G.add_edge(v0, v1, weight=dist)

    def _label_boundaries(self) -> None:
        """Detect boundary loops via face incidence counts and annotate nodes with boundary metadata."""
        try:
            edges = self.mesh.edges_unique
            counts = self.mesh.edges_unique_face_count
            boundary_edges = edges[counts == 1]
        except (AttributeError, ValueError):
            edge_count: Dict[Tuple[int, int], int] = {}
            for face in tqdm(
                self.mesh.faces,
                total=len(self.mesh.faces),
                disable=not self.show_progress,
                desc="Scanning faces for boundary edges",
                unit="face",
            ):
                for i in range(3):
                    u = int(face[i]); v = int(face[(i + 1) % 3])
                    key = tuple(sorted((u, v)))
                    edge_count[key] = edge_count.get(key, 0) + 1
            boundary_edges = np.array([list(e) for e, c in edge_count.items() if c == 1])

        Gb = nx.Graph()
        for u, v in tqdm(
            boundary_edges,
            total=len(boundary_edges),
            disable=not self.show_progress,
            desc="Assembling boundary graph",
            unit="edge",
        ):
            Gb.add_edge(int(u), int(v), weight=self.G[u][v]['weight'])

        comps = list(nx.connected_components(Gb))
        self.border_count = len(comps)

        loop_data = []
        for comp_idx, comp in enumerate(comps):
            zs = [self.G.nodes[n]['pos'][2] for n in comp]
            loop_data.append((comp_idx, sum(zs) / len(zs), comp))
        loop_data.sort(key=lambda t: t[1])

        for boundary_number, (_, _, comp) in enumerate(loop_data):
            for n in comp:
                self.G.nodes[n]['node_type'] = 'boundary'
                self.G.nodes[n]['boundary_number'] = boundary_number
        for n in self.G.nodes():
            if 'node_type' not in self.G.nodes[n]:
                self.G.nodes[n]['node_type'] = 'intra'
                self.G.nodes[n]['boundary_number'] = None

    # --------------------------------------------------------------------- #
    #                Boundary loops + neighbor ring (internal utils)        #
    # --------------------------------------------------------------------- #
    def _boundary_loops_data(self) -> Dict[int, Dict[str, Any]]:
        """Return cached metadata for each detected boundary loop.
        
        Returns
        -------
        Dict[int, Dict[str, Any]]
            Mapping from boundary id to a dictionary containing node sets, boundary edges,
            and geometric descriptors such as the centroid."""
        loops: Dict[int, Dict[str, Any]] = {}
        for n, d in self.G.nodes(data=True):
            if d.get('node_type') == 'boundary':
                b = int(d['boundary_number'])
                if b not in loops:
                    loops[b] = {'nodes': set(), 'edges': [], 'centroid': None, 'mean': {}}
                loops[b]['nodes'].add(int(n))

        if not loops:
            return {}

        try:
            edges = self.mesh.edges_unique
            counts = self.mesh.edges_unique_face_count
            boundary_edges = edges[counts == 1]
        except (AttributeError, ValueError):
            edge_count: Dict[Tuple[int, int], int] = {}
            for face in self.mesh.faces:
                for i in range(3):
                    u = int(face[i]); v = int(face[(i + 1) % 3])
                    key = tuple(sorted((u, v)))
                    edge_count[key] = edge_count.get(key, 0) + 1
            boundary_edges = np.array([list(e) for e, c in edge_count.items() if c == 1], dtype=int)

        for u, v in boundary_edges:
            du = self.G.nodes[int(u)]
            dv = self.G.nodes[int(v)]
            if du.get('node_type') == 'boundary' and dv.get('node_type') == 'boundary':
                bu = int(du['boundary_number'])
                bv = int(dv['boundary_number'])
                if bu == bv and bu in loops:
                    loops[bu]['edges'].append((int(u), int(v)))

        for b, info in loops.items():
            pts = np.array([self.G.nodes[n]['pos'] for n in info['nodes']], dtype=float)
            c = pts.mean(axis=0)
            info['centroid'] = (float(c[0]), float(c[1]), float(c[2]))
            info['mean'] = {'x': float(pts[:, 0].mean()),
                            'y': float(pts[:, 1].mean()),
                            'z': float(pts[:, 2].mean())}
        return loops

    # --------------------------------------------------------------------- #
    #                 Public helpers: distances and node utilities          #
    # --------------------------------------------------------------------- #
    def _ensure_csr_graph(self) -> None:
        """Construct CSR adjacency arrays used by the fast Dijkstra kernels if they have not already been built."""
        if self._csr_indptr is not None and self._csr_indices is not None and self._csr_weights is not None:
            return
        indptr = [0]
        indices: List[int] = []
        weights: List[float] = []
        for node in self._sorted_nodes:
            nbrs = self.G[int(node)]
            for nbr, data in nbrs.items():
                indices.append(self._node_index[int(nbr)])
                weights.append(float(data.get('weight', 1.0)))
            indptr.append(len(indices))
        self._csr_indptr = np.array(indptr, dtype=np.int64)
        self._csr_indices = np.array(indices, dtype=np.int64) if indices else np.empty(0, dtype=np.int64)
        self._csr_weights = np.array(weights, dtype=np.float64) if weights else np.empty(0, dtype=np.float64)

    def _single_source_distances(self, source: int) -> np.ndarray:
        """Compute geodesic distances from a single source vertex.
        
        Parameters
        ----------
        source : int
            Vertex id used as the seed.
        
        Returns
        -------
        np.ndarray
            Distance vector aligned with `_sorted_nodes`."""
        if HAS_NUMBA and source in self._node_index:
            self._ensure_csr_graph()
            if self._csr_indptr is not None and self._csr_indices is not None and self._csr_weights is not None:
                dist = dijkstra_single_source_jit(
                    self._csr_indptr,
                    self._csr_indices,
                    self._csr_weights,
                    self._node_index[int(source)],
                )
                if dist is not None:
                    return dist.copy()
        dmap = nx.single_source_dijkstra_path_length(self.G, source, weight='weight')
        vec = np.array([dmap.get(n, np.inf) for n in self._sorted_nodes], dtype=float)
        return vec

    def compute_axis_distance(self, axis: str) -> np.ndarray:
        """Return per-vertex coordinates along a principal axis with the minimum value shifted to zero."""
        axis = axis.lower()
        if axis not in {"x", "y", "z"}:
            raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        coords = self.mesh.vertices[:, idx]
        return coords - coords.min()

    def compute_geodesic_distance_from_boundary(self, boundary_number: int) -> np.ndarray:
        """Compute multi-source geodesic distances from a labelled boundary loop.
        
        Parameters
        ----------
        boundary_number : int
            Identifier assigned by :meth:_label_boundaries.
        
        Returns
        -------
        np.ndarray
            Distance vector with `np.inf` for vertices that are unreachable."""
        sources = [
            n for n, d in self.G.nodes(data=True)
            if d['node_type'] == 'boundary' and d['boundary_number'] == boundary_number
        ]
        if not sources:
            vec = np.full(len(self._sorted_nodes), np.inf, dtype=float)
        elif HAS_NUMBA:
            self._ensure_csr_graph()
            if self._csr_indptr is not None and self._csr_indices is not None and self._csr_weights is not None:
                try:
                    mapped = np.array([self._node_index[int(s)] for s in sources], dtype=np.int64)
                except KeyError:
                    mapped = None
                if mapped is not None:
                    dist = dijkstra_multi_source_jit(
                        self._csr_indptr,
                        self._csr_indices,
                        self._csr_weights,
                        mapped,
                    )
                    if dist is not None:
                        vec = dist.copy()
                    else:
                        dmap = nx.multi_source_dijkstra_path_length(self.G, sources, weight='weight')
                        vec = np.array([dmap.get(n, np.inf) for n in self._sorted_nodes], dtype=float)
                else:
                    dmap = nx.multi_source_dijkstra_path_length(self.G, sources, weight='weight')
                    vec = np.array([dmap.get(n, np.inf) for n in self._sorted_nodes], dtype=float)
            else:
                dmap = nx.multi_source_dijkstra_path_length(self.G, sources, weight='weight')
                vec = np.array([dmap.get(n, np.inf) for n in self._sorted_nodes], dtype=float)
        else:
            dmap = nx.multi_source_dijkstra_path_length(self.G, sources, weight='weight')
            vec = np.array([dmap.get(n, np.inf) for n in self._sorted_nodes], dtype=float)
        self.border_geodesic_distances[boundary_number] = vec
        return vec


    def get_boundary_vertices(self, side: Optional[str] = None) -> List[int]:
        """Return vertices lying on boundary loops, optionally filtered by side.

        Parameters
        ----------
        side : {'upper', 'lower', None}, optional
            Restrict the return value to a boundary side. When ``None`` (default) the
            method returns every boundary vertex detected on the mesh.

        Returns
        -------
        List[int]
            Sorted vertex indices that belong to the requested boundary set.

        Raises
        ------
        RuntimeError
            If the requested boundary side is unavailable.
        """
        loops = self._boundary_loops_data()
        if not loops:
            return []

        if side is None:
            nodes = set()
            for data in loops.values():
                nodes.update(int(v) for v in data.get('nodes', ()))
            return sorted(nodes)

        side_l = str(side).lower()
        if side_l not in ('upper', 'lower'):
            raise ValueError("side must be 'upper', 'lower', or None")

        if not hasattr(self, 'boundary_sides') or not getattr(self, 'boundary_sides', None):
            raise RuntimeError("Boundary sides are missing. Call `label_lower_upper_boundaries(...)` first.")

        selected = [int(b) for b, s in self.boundary_sides.items() if s == side_l]
        if not selected:
            raise RuntimeError(f"No boundary loops classified as '{side_l}'.")

        nodes = set()
        for b in selected:
            data = loops.get(int(b))
            if data:
                nodes.update(int(v) for v in data.get('nodes', ()))

        if not nodes:
            raise RuntimeError(f"No boundary vertices found for side '{side_l}'.")

        return sorted(nodes)

    def compute_min_distance_to_boundary_side(self, side: str) -> np.ndarray:
        """Return per-vertex geodesic distance to the closest boundary on ``side``.

        Parameters
        ----------
        side : {'upper', 'lower'}
            Boundary side for which the minimum distance should be reported.
        """
        side_l = str(side).lower()
        if side_l not in ('upper', 'lower'):
            raise ValueError("side must be 'upper' or 'lower'")

        if not hasattr(self, 'boundary_sides') or not getattr(self, 'boundary_sides', None):
            raise RuntimeError("Boundary sides are missing. Call `label_lower_upper_boundaries(...)` first.")

        loops = self._boundary_loops_data()
        if not loops:
            raise RuntimeError("No boundary loops found on the mesh.")

        side_boundaries = [int(b) for b, s in self.boundary_sides.items() if s == side_l]
        if not side_boundaries:
            raise RuntimeError(f"No boundary loops classified as '{side_l}'.")

        dists: List[np.ndarray] = []
        for boundary_id in side_boundaries:
            dist = self.compute_geodesic_distance_from_boundary(boundary_id)
            dists.append(np.asarray(dist, dtype=float))

        if not dists:
            return np.full(len(self.mesh.vertices), np.inf, dtype=float)

        D = np.vstack(dists).T
        return np.min(D, axis=1)

    def compute_min_distance_to_saddles(self, saddle_vertices: Optional[Sequence[int]] = None) -> np.ndarray:
        """Return per-vertex geodesic distance to the closest saddle vertex.

        Parameters
        ----------
        saddle_vertices : Sequence[int], optional
            Explicit list of saddle vertex ids. When omitted the method relies on
            previously annotated saddles stored on the graph.
        """
        seeds: List[int]
        if saddle_vertices is None:
            seeds = []
        else:
            seeds = [int(s) for s in saddle_vertices if s is not None]

        if not seeds:
            annotated = [
                int(n)
                for n, data in self.G.nodes(data=True)
                if data.get('morse_index') == 1 or data.get('critical_type') == 'saddle'
            ]
            seeds = sorted(set(annotated))

        if not seeds:
            return np.full(len(self.mesh.vertices), np.inf, dtype=float)

        distances = self.compute_geodesic_distance_to_nodes(seeds)
        return np.asarray(distances, dtype=float)

    def compute_geodesic_distance_to_nodes(self, nodes_list: Sequence[int]) -> np.ndarray:
        """Compute geodesic distances to a collection of source vertices.
        
        Parameters
        ----------
        nodes_list : Sequence[int]
            Node ids used as distance seeds.
        
        Returns
        -------
        np.ndarray
            Distance vector aligned with `_sorted_nodes`."""
        if not nodes_list:
            vec = np.full(len(self._sorted_nodes), np.inf, dtype=float)
        elif HAS_NUMBA:
            self._ensure_csr_graph()
            if self._csr_indptr is not None and self._csr_indices is not None and self._csr_weights is not None:
                try:
                    mapped = np.array([self._node_index[int(n)] for n in nodes_list], dtype=np.int64)
                except KeyError:
                    mapped = None
                if mapped is not None:
                    dist = dijkstra_multi_source_jit(
                        self._csr_indptr,
                        self._csr_indices,
                        self._csr_weights,
                        mapped,
                    )
                    if dist is not None:
                        vec = dist.copy()
                    else:
                        dmap = nx.multi_source_dijkstra_path_length(self.G, nodes_list, weight='weight')
                        vec = np.array([dmap.get(n, np.inf) for n in self._sorted_nodes], dtype=float)
                else:
                    dmap = nx.multi_source_dijkstra_path_length(self.G, nodes_list, weight='weight')
                    vec = np.array([dmap.get(n, np.inf) for n in self._sorted_nodes], dtype=float)
            else:
                dmap = nx.multi_source_dijkstra_path_length(self.G, nodes_list, weight='weight')
                vec = np.array([dmap.get(n, np.inf) for n in self._sorted_nodes], dtype=float)
        else:
            dmap = nx.multi_source_dijkstra_path_length(self.G, nodes_list, weight='weight')
            vec = np.array([dmap.get(n, np.inf) for n in self._sorted_nodes], dtype=float)
        self.custom_geodesic_distances[tuple(sorted(int(n) for n in nodes_list))] = vec
        return vec

    def get_closest_nodes_from_coords(
        self, coords_list: Sequence[Tuple[float, float, float]]
    ) -> List[int]:
        """For each 3D coordinate, return the id of the nearest graph node.
        
        Parameters
        ----------
        coords_list : Sequence[Tuple[float, float, float]]
            Points to map onto the graph.
        
        Returns
        -------
        List[int]
            Closest node id per input coordinate."""
        node_ids = sorted(self.G.nodes())
        positions = np.array([self.G.nodes[n]['pos'] for n in node_ids], dtype=float)
        closest = []
        for coord in tqdm(
            coords_list,
            total=len(coords_list),
            disable=not self.show_progress,
            desc="Nearest node lookup",
            unit="pt",
        ):
            dists = np.linalg.norm(positions - np.array(coord, dtype=float), axis=1)
            closest.append(node_ids[int(np.argmin(dists))])
        return closest

    def save_graphml(self, filepath: str = 'mesh_graph.graphml') -> None:
        """Export the graph as GraphML, coercing non-serialisable attributes to strings on the fly.

        Parameters
        ----------
        filepath : str, optional
            Destination path for the GraphML file.
        """
        H = self.G.copy()
        # Nodes
        for n, data in tqdm(
            list(H.nodes(data=True)),
            total=H.number_of_nodes(),
            disable=not self.show_progress,
            desc="Preparing nodes for GraphML",
            unit="node",
        ):
            for k, v in list(data.items()):
                if v is None:
                    H.nodes[n][k] = "None"
                elif isinstance(v, tuple):
                    H.nodes[n][k] = str(v)
        # Edges
        for u, v, data in tqdm(
            list(H.edges(data=True)),
            total=H.number_of_edges(),
            disable=not self.show_progress,
            desc="Preparing edges for GraphML",
            unit="edge",
        ):
            for k, val in list(data.items()):
                if val is None:
                    H[u][v][k] = "None"
                elif isinstance(val, tuple):
                    H[u][v][k] = str(val)
        nx.write_graphml(H, filepath)

    # --------------------------------------------------------------------- #
    #                       Critical point detection (wrapped)              #
    # --------------------------------------------------------------------- #
    def detect_critical_points(
        self,
        scalar_field: np.ndarray,
        *,
        eps: float = 1e-10,
        clip_saddles_geodesic_threshold: float = 0.0,
        clip_saddles_strategy: str = 'confidence',
        resolve_plateaus: bool = True,
        normalize: bool = True,
        annotate: bool = True,
        smooth_field: bool = False,
        smooth_iterations: int = 2,
        adaptive_eps: bool = True,
        compute_confidence: bool = True,
        persistence_threshold: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Classify vertices of ``scalar_field`` into minima, saddles, maxima, and regular points.

Parameters
----------
scalar_field : np.ndarray
    Scalar field sampled per vertex.
eps : float, optional
    Global epsilon tolerance for the sign-change test.
clip_saddles_geodesic_threshold : float, optional
    Distance threshold used when merging nearby saddles.
clip_saddles_strategy : str, optional
    Strategy for ranking saddles during clustering.
resolve_plateaus, normalize, annotate, smooth_field, adaptive_eps, compute_confidence : bool, optional
    Behavioural flags forwarded to :func:`slicing.critical_points.classify_critical_points`.
smooth_iterations : int, optional
    Iteration count for Laplacian smoothing when enabled.
persistence_threshold : float, optional
    Minimum persistence required to keep a critical point.

Returns
-------
Dict[str, np.ndarray]
    Classification results and, when requested, auxiliary data such as confidence
    scores or persistence estimates.
"""
        faces = np.asarray(self.mesh.faces, dtype=int)
        # NOTE: do NOT pass a 'progress' kwarg here; keeps compatibility with both versions
        res = classify_critical_points(
            scalar_field=scalar_field,
            faces=faces,
            vertices=np.asarray(self.mesh.vertices, dtype=float),
            eps=eps,
            resolve_plateaus=resolve_plateaus,
            normalize=normalize,
            smooth_field=smooth_field,
            smooth_iterations=smooth_iterations,
            adaptive_eps=adaptive_eps,
            compute_confidence=compute_confidence,
            persistence_threshold=persistence_threshold,
            clip_saddles_geodesic_threshold=clip_saddles_geodesic_threshold,
            clip_saddles_strategy=clip_saddles_strategy,
        )

        if annotate:
            idx_map: Dict[int, int] = {}
            confidence_map = res.get('confidence', {})

            for v in res['minima']: idx_map[int(v)] = 0
            for v in res['saddles']: idx_map[int(v)] = 1
            for v in res['maxima']: idx_map[int(v)] = 2
            for v in res['regular']: idx_map[int(v)] = -1

            for v, d in self.G.nodes(data=True):
                mi = idx_map.get(v, -1)
                d['morse_index'] = mi
                d['critical_type'] = {0: 'minimum', 1: 'saddle', 2: 'maximum', -1: 'regular'}[mi]
                if v in confidence_map:
                    d['critical_confidence'] = float(confidence_map[v])

        return res

    def detect_saddle_points(
        self,
        scalar_field: np.ndarray,
        *,
        annotate: bool = False,
        multi_scale: bool = False,
        scale_levels: int = 3,
        min_consensus: float = 0.6,
        **kwargs: Any
    ) -> np.ndarray:
        """Detect saddle vertices in `scalar_field` and optionally annotate them on the graph.
        
        Parameters
        ----------
        scalar_field : np.ndarray
            Scalar field sampled per vertex.
        annotate : bool, optional
            Persist saddle annotations onto node attributes.
        
        Returns
        -------
        np.ndarray
            Sorted array of saddle vertex indices."""
        if not multi_scale:
            # Single-scale detection
            res = self.detect_critical_points(
                scalar_field,
                annotate=annotate,
                **kwargs
            )
            return res['saddles']

        # Multi-scale detection with consensus
        all_saddles = []
        vote_counts = defaultdict(int)

        # Base parameters
        base_smooth = kwargs.pop('smooth_iterations', 2)
        base_persist = kwargs.pop('persistence_threshold', 0.0)
        base_smooth_flag = kwargs.pop('smooth_field', None)
        apply_smoothing = True if base_smooth_flag is None else bool(base_smooth_flag)

        for level in tqdm(
            range(scale_levels),
            disable=not self.show_progress,
            desc="Multi-scale saddle detection",
            unit="scale",
        ):
            # Increase smoothing and persistence threshold at each scale
            smooth_iter = base_smooth * (level + 1)
            persist_thr = base_persist + (level * 0.01)

            res = self.detect_critical_points(
                scalar_field,
                annotate=False,
                smooth_field=apply_smoothing,
                smooth_iterations=smooth_iter,
                persistence_threshold=persist_thr,
                **kwargs
            )

            saddles = res['saddles']
            all_saddles.append(set(saddles))
            for s in saddles:
                vote_counts[s] += 1

        # Keep saddles that appear in sufficient scales
        min_votes = int(np.ceil(scale_levels * min_consensus))
        consensus_saddles = np.array([
            s for s, votes in vote_counts.items()
            if votes >= min_votes
        ], dtype=int)

        # Sort for consistency
        consensus_saddles = np.sort(consensus_saddles)

        # Annotate if requested
        if annotate:
            _ = self.detect_critical_points(
                scalar_field,
                annotate=True,
                **kwargs
            )

        return consensus_saddles

    def validate_critical_points(
        self,
        scalar_field: np.ndarray,
        critical_points: Dict[str, np.ndarray],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Re-evaluate the classification for the supplied critical sets and report mismatches."""
        n_min = len(critical_points.get('minima', []))
        n_max = len(critical_points.get('maxima', []))
        n_sad = len(critical_points.get('saddles', []))

        # Compute Euler characteristic
        V = len(self.mesh.vertices)
        E = len(self.mesh.edges_unique)
        F = len(self.mesh.faces)
        euler = V - E + F

        # For closed surfaces, should have: #max - #saddle + #min = euler
        morse_euler = n_max - n_sad + n_min

        validation = {
            'euler_characteristic': euler,
            'morse_euler': morse_euler,
            'euler_error': abs(euler - morse_euler),
            'n_minima': n_min,
            'n_maxima': n_max,
            'n_saddles': n_sad,
            'is_valid': abs(euler - morse_euler) <= 2,  # Allow small error
        }

        if verbose:
            print(f"Critical Point Validation:")
            print(f"  Minima: {n_min}, Maxima: {n_max}, Saddles: {n_sad}")
            print(f"  Euler characteristic: {euler}")
            print(f"  Morse-Euler sum: {morse_euler}")
            print(f"  Valid: {'OK' if validation['is_valid'] else 'NO'} (error: {validation['euler_error']})")

        return validation

    # --------------------------------------------------------------------- #
    #                   Lower/Upper boundary labeling + nearest             #
    # --------------------------------------------------------------------- #
    def label_lower_upper_boundaries(
        self,
        *,
        axis: str = 'z',
        side_tol: float = 0.05,
        assign_middle_to_nearest: bool = True,
        overwrite: bool = True,
    ) -> Dict[int, str]:
        """Assign each boundary loop to the lower or upper side along a chosen axis.
        
        Parameters
        ----------
        axis : str, optional
            Axis used to determine ordering ('x', 'y', or 'z').
        side_tol : float, optional
            Fraction of the axis range used to classify loops near the extrema.
        assign_middle_to_nearest : bool, optional
            If `True` classify middle loops by proximity to the extremes; otherwise leave them unlabelled.
        overwrite : bool, optional
            When `False` preserve existing assignments."""
        axis = axis.lower()
        if axis not in ('x', 'y', 'z'):
            raise ValueError("axis must be 'x','y' or 'z'")

        loops = self._boundary_loops_data()
        if not loops:
            self.boundary_sides = {}
            return {}

        means = {b: info['mean'][axis] for b, info in loops.items()}
        vals = np.array(list(means.values()), dtype=float)
        bnums = list(means.keys())
        vmin, vmax = float(vals.min()), float(vals.max())
        rng = max(1e-14, vmax - vmin)

        lower_thr = vmin + side_tol * rng
        upper_thr = vmax - side_tol * rng

        side_map: Dict[int, str] = {}
        for b, m in means.items():
            if m <= lower_thr:
                side_map[b] = 'lower'
            elif m >= upper_thr:
                side_map[b] = 'upper'
        if assign_middle_to_nearest:
            for b in bnums:
                if b not in side_map:
                    dl = abs(means[b] - vmin); du = abs(means[b] - vmax)
                    side_map[b] = 'lower' if dl <= du else 'upper'

        # Mark edges and nodes with progress
        total_edges = sum(len(info['edges']) for info in loops.values())
        total_nodes = sum(len(info['nodes']) for info in loops.values())

        if overwrite:
            for u, v in self.G.edges():
                self.G[u][v]['edge_type'] = 'internal'

        with tqdm(
            total=total_edges,
            disable=not self.show_progress,
            desc="Stamping boundary edges",
            unit="edge",
        ) as pbar_e:
            for b, info in loops.items():
                side = side_map[b]
                for u, v in info['edges']:
                    self.G[u][v]['edge_type'] = 'boundary'
                    self.G[u][v]['boundary_number'] = b
                    if overwrite or ('boundary_side' not in self.G[u][v]):
                        self.G[u][v]['boundary_side'] = side
                    pbar_e.update(1)

        with tqdm(
            total=total_nodes,
            disable=not self.show_progress,
            desc="Stamping boundary nodes",
            unit="node",
        ) as pbar_n:
            for b, info in loops.items():
                side = side_map[b]
                for n in info['nodes']:
                    if overwrite or ('boundary_side' not in self.G.nodes[n]):
                        self.G.nodes[n]['boundary_side'] = side
                    pbar_n.update(1)

        self.boundary_sides = side_map
        return side_map

    def assign_nearest_lower_upper_boundaries(
        self,
        *,
        axis: str = 'z',
        side_tol: float = 0.05
    ) -> None:
        """Propagate boundary-side labels to interior nodes by nearest geodesic distance.
        
        Parameters
        ----------
        axis : str, optional
            Axis used when resolving ambiguities for middle loops.
        side_tol : float, optional
            Tolerance for recognising upper and lower boundary loops."""
        if not hasattr(self, 'boundary_sides') or not self.boundary_sides:
            self.label_lower_upper_boundaries(axis=axis, side_tol=side_tol)

        loops = self._boundary_loops_data()
        if not loops:
            for n in self.G.nodes():
                self.G.nodes[n]['closest_lower_boundary'] = None
                self.G.nodes[n]['closest_upper_boundary'] = None
                self.G.nodes[n]['dist_to_lower'] = float('inf')
                self.G.nodes[n]['dist_to_upper'] = float('inf')
            return

        lower_b = [b for b, s in self.boundary_sides.items() if s == 'lower']
        upper_b = [b for b, s in self.boundary_sides.items() if s == 'upper']

        node_ids = sorted(self.G.nodes())
        N = len(node_ids)

        if lower_b:
            dmat_lower = np.vstack([self.compute_geodesic_distance_from_boundary(b) for b in lower_b]).T
            idx_lower = np.argmin(dmat_lower, axis=1)
            best_lower = np.take(lower_b, idx_lower)
            dist_lower = dmat_lower[np.arange(N), idx_lower]
        else:
            best_lower = np.array([None] * N, dtype=object)
            dist_lower = np.full(N, np.inf, dtype=float)

        if upper_b:
            dmat_upper = np.vstack([self.compute_geodesic_distance_from_boundary(b) for b in upper_b]).T
            idx_upper = np.argmin(dmat_upper, axis=1)
            best_upper = np.take(upper_b, idx_upper)
            dist_upper = dmat_upper[np.arange(N), idx_upper]
        else:
            best_upper = np.array([None] * N, dtype=object)
            dist_upper = np.full(N, np.inf, dtype=float)

        for i, n in enumerate(tqdm(
            node_ids,
            total=N,
            disable=not self.show_progress,
            desc="Assigning nearest boundaries",
            unit="node",
        )):
            self.G.nodes[n]['closest_lower_boundary'] = None if best_lower[i] is None else int(best_lower[i])
            self.G.nodes[n]['closest_upper_boundary'] = None if best_upper[i] is None else int(best_upper[i])
            self.G.nodes[n]['dist_to_lower'] = float(dist_lower[i])
            self.G.nodes[n]['dist_to_upper'] = float(dist_upper[i])

    # --------------------------------------------------------------------- #
    #                    Geometry-conforming scalar field (PDF)             #
    # --------------------------------------------------------------------- #
    def compute_conforming_scalar_field(
        self,
        *,
        axis_for_boundary: str = 'z',
        saddle_vertices: Optional[Sequence[int]] = None,
        saddle_field: Optional[np.ndarray] = None,
        # Weight parameters controlling saddle influence radii
        radii_per_saddle: Optional[Sequence[float]] = None,
        n: float = 2.0,
        eps: float = 1e-9,
        chunk_progress: bool = True,
    ) -> np.ndarray:
        """Construct a scalar field `F` in `[0, 1]` that honours boundary values and saddle constraints.
        
        The method combines geodesic distances from vertices to each boundary loop with
        saddle-centric weights. For every saddle `s` a radius `R_s` is estimated (or
        provided) and weights `W_vs` are computed using `csch(sqrt(max(D_vs + R_s^n - R_s, 0)))`.
        The weighted biases shift boundary distances before the final blend
        
            F = R_upper / (R_upper + R_lower + eps)
        
        where `R_upper` and `R_lower` are the biased distances to the respective
        boundary sets. The resulting field is written back onto node attributes as
        `conforming_scalar`.
        
        Parameters
        ----------
        axis_for_boundary : str, optional
            Axis used when classifying boundary loops into lower/upper sets.
        saddle_vertices : Optional[Sequence[int]]
            Pre-computed saddle vertex indices; detected automatically when omitted.
        saddle_field : Optional[np.ndarray]
            Scalar field used when auto-detecting saddles.
        radii_per_saddle : Optional[Sequence[float]]
            Override for the saddle influence radii.
        n : float, optional
            Exponent controlling the radial fall-off around saddles.
        eps : float, optional
            Numerical guard used in the final blend.
        chunk_progress : bool, optional
            Show progress bars while computing geodesic tables.
        
        Returns
        -------
        np.ndarray
            Conforming scalar field normalised to `[0, 1]`."""
        V = len(self.mesh.vertices)
        node_ids = sorted(self.G.nodes())

        # Ensure boundary sides for nodes/edges
        self.label_lower_upper_boundaries(axis=axis_for_boundary)

        # Boundary vertices by side (vertices on boundary edges)
        b_upper = [n for n in node_ids if self.G.nodes[n].get('node_type') == 'boundary'
                   and self.G.nodes[n].get('boundary_side') == 'upper']
        b_lower = [n for n in node_ids if self.G.nodes[n].get('node_type') == 'boundary'
                   and self.G.nodes[n].get('boundary_side') == 'lower']

        N_up, N_lo = len(b_upper), len(b_lower)
        if N_up == 0 or N_lo == 0:
            raise RuntimeError("Both upper and lower boundary vertices are required to compute F.")

        # --- Find saddles (S)
        if saddle_vertices is None:
            field_for_saddles = saddle_field if saddle_field is not None else self.compute_axis_distance(axis_for_boundary)
            saddles = self.detect_saddle_points(field_for_saddles, annotate=True)
        else:
            saddles = np.array([int(s) for s in saddle_vertices], dtype=int)

        S = int(saddles.size)

        # ---------- Step 1: D_vs (VxS) and D_sn (SxNx2) ----------
        if S > 0:
            D_vs = np.zeros((V, S), dtype=float)
            D_sn_up = np.zeros((S, N_up), dtype=float)
            D_sn_lo = np.zeros((S, N_lo), dtype=float)

            iter_saddles = tqdm(
                range(S),
                desc="Distances from each saddle",
                unit="saddle",
                disable=not (chunk_progress and self.show_progress)
            )
            for si in iter_saddles:
                s = int(saddles[si])
                ds_all = self._single_source_distances(s)  # distances from saddle s to all vertices
                D_vs[:, si] = ds_all
                # pick at boundary vertices for D_sn
                D_sn_up[si, :] = ds_all[np.array(b_upper, dtype=int)]
                D_sn_lo[si, :] = ds_all[np.array(b_lower, dtype=int)]
        else:
            D_vs = np.zeros((V, 0), dtype=float)
            D_sn_up = np.zeros((0, N_up), dtype=float)
            D_sn_lo = np.zeros((0, N_lo), dtype=float)

        # ---------- Step 2: W (VxS) weights via sqrt + csch ----------
        if S > 0:
            if radii_per_saddle is None:
                R_up_min = D_sn_up.min(axis=1) if N_up > 0 else np.full(S, np.inf)
                R_lo_min = D_sn_lo.min(axis=1) if N_lo > 0 else np.full(S, np.inf)
                R = np.minimum(R_up_min, R_lo_min)  # (S,)
            else:
                R = np.asarray(radii_per_saddle, dtype=float).reshape(-1)
                if R.size != S:
                    raise ValueError("radii_per_saddle must have length equal to number of saddles S.")

            X = np.maximum(D_vs + (R ** n)[None, :] - R[None, :], 0.0)
            X = np.sqrt(X)
            W_raw = 1.0 / np.sinh(np.maximum(X, 1e-12))
            denom = W_raw.sum(axis=1, keepdims=True) + 1e-12
            W = W_raw / denom  # (V,S)
        else:
            W = np.zeros((V, 0), dtype=float)

        # ---------- Step 3: D_vn (VxNx2) and B = W x D_sn ----------
        def build_Dvn(boundary_list: List[int], name: str) -> np.ndarray:
            Nb = len(boundary_list)
            D = np.zeros((V, Nb), dtype=float)
            iterator = tqdm(
                range(Nb),
                desc=f"D_vn to {name} boundary",
                unit="src",
                disable=not (chunk_progress and self.show_progress),
            )
            for j in iterator:
                bj = int(boundary_list[j])
                D[:, j] = self._single_source_distances(bj)
            return D  # shape (V, Nb)

        D_vn_up = build_Dvn(b_upper, "upper")   # (V, N_up)
        D_vn_lo = build_Dvn(b_lower, "lower")   # (V, N_lo)

        # Bias via broadcasted matrix multiplication (per side):
        if S > 0:
            B_up = W @ D_sn_up
            B_lo = W @ D_sn_lo
        else:
            B_up = np.zeros_like(D_vn_up)
            B_lo = np.zeros_like(D_vn_lo)

        T_up = D_vn_up + B_up
        T_lo = D_vn_lo + B_lo

        # ---------- Step 4: Reductions and psi ----------
        Rupper = T_up.min(axis=1)
        Rlower = T_lo.min(axis=1)
        F = Rupper / (Rupper + Rlower + float(eps))

        # Attach for convenience
        for i, n in enumerate(node_ids):
            self.G.nodes[n]['conforming_scalar'] = float(F[i])

        return F

    # --------------------------------------------------------------------- #
    #                          Iso-slicing: Helpers                         #
    # --------------------------------------------------------------------- #
    def _normalize_field(self, field: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """Normalise a scalar field to `[0, 1]` and return the scaling statistics.
        
        Parameters
        ----------
        field : np.ndarray
            Scalar field sampled per vertex.
        
        Returns
        -------
        Tuple[np.ndarray, float, float, float]
            Normalised field together with `(vmin, vmax, range)`."""
        vals = np.asarray(field, dtype=float)
        if vals.ndim != 1 or vals.size != len(self.mesh.vertices):
            raise ValueError("scalar_field must be shape (V,) and match mesh vertices.")
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            raise ValueError("scalar_field contains non-finite values.")
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        rng = vmax - vmin
        if rng <= 0:
            r = np.zeros_like(vals)
        else:
            r = (vals - vmin) / rng
            r = np.clip(r, 0.0, 1.0)
        return r, vmin, vmax, rng

    def _compute_face_grad_norms(self, r: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Estimate per-face gradient magnitudes using the JIT kernel when available, falling back to Python otherwise.\n\n        Parameters\n        ----------\n        r : np.ndarray\n            Normalised scalar field sampled per vertex.\n        eps : float, optional\n            Numerical guard for degenerate triangles.\n\n        Returns\n        -------\n        np.ndarray\n            Gradient magnitude for every face."""
        verts = np.asarray(self.mesh.vertices, dtype=np.float64)
        faces = np.asarray(self.mesh.faces, dtype=np.int64)
        values = np.asarray(r, dtype=np.float64).reshape(-1)
        if values.ndim != 1 or values.shape[0] != verts.shape[0]:
            raise ValueError("scalar field must align with vertex count")

        jit_result = compute_face_grad_norms_jit(verts, faces, values, float(eps))
        if jit_result is not None:
            return jit_result

        gn = np.zeros(faces.shape[0], dtype=float)
        iterator = tqdm(
            faces,
            total=faces.shape[0],
            disable=not self.show_progress,
            desc="Per-face gradient norms (fallback)",
            unit="face",
        )
        for fi, (i, j, k) in enumerate(iterator):
            pi, pj, pk = verts[i], verts[j], verts[k]
            ri, rj, rk = float(values[i]), float(values[j]), float(values[k])

            e1 = pj - pi
            e2 = pk - pi
            n = np.cross(e1, e2)
            nlen = np.linalg.norm(n)
            ulen = np.linalg.norm(e1)
            if nlen < eps or ulen < eps:
                gn[fi] = 0.0
                continue

            uhat = e1 / ulen
            vtemp = e2 - np.dot(e2, uhat) * uhat
            vlen = np.linalg.norm(vtemp)
            if vlen < eps:
                gn[fi] = 0.0
                continue
            vhat = vtemp / vlen

            xj, yj = ulen, 0.0
            xk = float(np.dot(e2, uhat))
            yk = float(np.dot(e2, vhat))

            if abs(xj) < eps or abs(yk) < eps:
                gn[fi] = 0.0
                continue

            a = (rj - ri) / xj
            b = (rk - ri - xk * a) / yk
            gn[fi] = float(np.hypot(a, b))
        return gn

    def _build_slice_at_level(
        self,
        level: float,
        r: np.ndarray,
        gn_face: np.ndarray,
        *,
        orientation_hint: np.ndarray | None,
        deg: int,
        samples: int,
        periodic: bool,
        dedupe_decimals: int,
        edge_tol: float,
        verbose: bool,
        min_component_points: int,
    ) -> Tuple[Optional[IsoSlice], int, float, float, float]:
        """Intersect the scalar field with iso-level `level` and assemble an :class:`IsoSlice`.
        
        Parameters
        ----------
        orientation_hint : np.ndarray | None
            Optional global direction used to enforce consistent curve orientation.

        Returns
        -------
        Tuple[Optional[IsoSlice], int, float, float, float]
            The constructed slice (or `None` when no geometry is found), number of raw
            points, accumulated polyline length, gradient-weighted length, and the final
            curve length."""
        verts64 = np.asarray(self.mesh.vertices, dtype=np.float64)
        faces64 = np.asarray(self.mesh.faces, dtype=np.int64)
        values = np.asarray(r, dtype=np.float64)
        grad_norms = np.asarray(gn_face, dtype=np.float64)

        segments: List[Tuple[np.ndarray, np.ndarray]] = []
        length_sum = 0.0
        weighted_grad_sum = 0.0

        jit_data = extract_iso_segments_jit(
            verts64,
            faces64,
            values,
            float(level),
            float(edge_tol),
            grad_norms,
        )

        if jit_data is not None:
            seg_start, seg_end, mask, lengths, weighted = jit_data
            mask_bool = mask.astype(bool)
            for idx in range(mask_bool.shape[0]):
                if mask_bool[idx]:
                    segments.append((seg_start[idx].copy(), seg_end[idx].copy()))
            length_sum = float(np.sum(lengths))
            weighted_grad_sum = float(np.sum(weighted))
        else:
            iterator = tqdm(
                faces64,
                total=faces64.shape[0],
                disable=not self.show_progress,
                desc=f"Iso build @ level={level:.6g} (fallback)",
                unit="face",
                leave=False,
            )
            for fi, face in enumerate(iterator):
                fvals = values[face]
                pts: List[np.ndarray] = []

                for a, b in ((0, 1), (1, 2), (2, 0)):
                    fa = float(fvals[a])
                    fb = float(fvals[b])
                    da, db = fa - level, fb - level
                    if (da > edge_tol and db > edge_tol) or (da < -edge_tol and db < -edge_tol):
                        continue
                    denom = fb - fa
                    if abs(denom) < edge_tol:
                        continue
                    t = (level - fa) / denom
                    if t < -edge_tol or t > 1.0 + edge_tol:
                        continue
                    p0, p1 = verts64[face[a]], verts64[face[b]]
                    pts.append(p0 + t * (p1 - p0))

                if len(pts) == 2:
                    p0 = np.asarray(pts[0], dtype=np.float64)
                    p1 = np.asarray(pts[1], dtype=np.float64)
                    segments.append((p0, p1))
                    seg_len = float(np.linalg.norm(p1 - p0))
                    length_sum += seg_len
                    g = grad_norms[fi]
                    if np.isfinite(g):
                        weighted_grad_sum += g * seg_len

        components = build_slice_components(
            level,
            segments,
            min_points=max(2, min_component_points),
            rounding_decimals=dedupe_decimals,
            orientation_hint=orientation_hint,
        )
        if not components:
            return None, 0, length_sum, weighted_grad_sum, 0.0

        component_points = [comp.points for comp in components]
        component_lengths = [comp.curve_length for comp in components]
        primary = max(components, key=lambda comp: comp.points.shape[0])

        slice_obj = IsoSlice(
            level=level,
            points=primary.points,
            degree=deg,
            samples=samples,
            periodic=periodic,
            components=component_points,
            component_lengths=component_lengths,
            points_ordered=True,
            show_progress=self.show_progress,
        )

        point_count = sum(comp.points.shape[0] for comp in components)
        curve_len = slice_obj.total_length
        return slice_obj, point_count, length_sum, weighted_grad_sum, curve_len

    def _choose_intervals_and_hmod(
        self,
        h_in: float,
        g_eff: float,
        dr_clip: Tuple[float, float],
    ) -> Tuple[int, float, float]:
        """Derive the number of iso-intervals and the modified layer height used by the adaptive controller."""
        if not np.isfinite(h_in) or h_in <= 0:
            raise ValueError("layer_height must be a positive finite number.")
        step_norm_target = h_in * g_eff
        step_norm_target = float(np.clip(
            (dr_clip[0] if not np.isfinite(step_norm_target) or step_norm_target <= 0 else step_norm_target),
            dr_clip[0], dr_clip[1]
        ))
        N_intervals = max(1, int(round(1.0 / step_norm_target)))
        step_norm_mod = 1.0 / N_intervals
        h_mod = step_norm_mod / g_eff
        return N_intervals, step_norm_mod, h_mod


    def _infer_orientation_hint(self, values: np.ndarray) -> np.ndarray:
        """Infer a dominant direction for enforcing consistent curve orientation."""
        pts = np.asarray(self.mesh.vertices, dtype=float)
        vals = np.asarray(values, dtype=float).reshape(-1)
        if pts.shape[0] < 3 or vals.shape[0] != pts.shape[0]:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        A = np.hstack([pts, np.ones((pts.shape[0], 1))])
        try:
            sol, *_ = np.linalg.lstsq(A, vals, rcond=None)
            direction = sol[:3]
        except np.linalg.LinAlgError:
            direction = np.zeros(3, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            centered = pts - pts.mean(axis=0)
            cov = np.cov(centered, rowvar=False)
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                return np.array([0.0, 0.0, 1.0], dtype=float)
            direction = eigvecs[:, int(np.argmax(eigvals))]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9:
                return np.array([0.0, 0.0, 1.0], dtype=float)
        return (direction / norm).astype(float)

    def _init_progress_bar(
        self,
        total_slices: int,
        h_in: float,
        h_mod: float,
        progress_bar_width: int,
    ) -> tqdm:
        """Initialise a tqdm progress bar tailored to the adaptive slicing run."""
        desc = f"Adaptive slicing | h_in={h_in:.6g} mm | h_mod={h_mod:.6g} mm"
        bar_fmt = (
            "{desc} | {percentage:3.0f}% | slice {n}/{total} || "
            f"{{bar:{progress_bar_width}}} | {{elapsed}}<{{remaining}}, {{rate_fmt}} {{postfix}}"
        )
        return tqdm(
            total=total_slices,
            bar_format=bar_fmt,
            unit="slice",
            desc=desc,
            dynamic_ncols=True,     # keeps to one line even as console width changes
            mininterval=0.15,       # slightly throttles redraws to reduce flicker
            leave=True,
            disable=not self.show_progress,
        )

    # --------------------------------------------------------------------- #
    #                       Iso-slicing: Public Method                       #
    # --------------------------------------------------------------------- #
    def extract_iso_slices(
        self,
        scalar_field: np.ndarray,
        layer_height: float,
        periodic: bool = True,
        degree: int = 3,
        samples: int = 200,
        verbose: bool = False,
        *,
        dedupe_decimals: int = 6,
        edge_tol: float = 1e-6,
        dr_clip: Tuple[float, float] = (1e-4, 0.2),
        max_levels: int = 100000,
        include_end: bool = True,
        progress_bar_width: int = 80,
        controller_blend: float = 0.5,
        min_component_points: int = 3,
    ) -> "IsoSliceCollection":
        """Extract an :class:`IsoSliceCollection` from a normalised scalar field using adaptive spacing.
        
        Parameters
        ----------
        scalar_field : np.ndarray
            Scalar field sampled per vertex.
        layer_height : float
            Requested physical spacing between consecutive slices.
        periodic, degree, samples : optional
            Parameters forwarded to :class:`IsoSlice` for curve reconstruction.
        controller_blend : float, optional
            Blend factor between remaining-range and gradient-driven step sizes.
        
        Returns
        -------
        IsoSliceCollection
            Ordered collection of iso-slices covering `[0, 1]` within the configured tolerance."""
        if layer_height <= 0 or not np.isfinite(layer_height):
            raise ValueError("layer_height must be a positive finite number.")

        r, vmin, vmax, rng = self._normalize_field(scalar_field)
        orientation_hint = self._infer_orientation_hint(r)
        gn_face = self._compute_face_grad_norms(r)
        gn_finite = gn_face[np.isfinite(gn_face) & (gn_face >= 0)]
        g_eff = float(np.median(gn_finite)) if gn_finite.size else 0.0
        if g_eff <= 0 or not np.isfinite(g_eff):
            g_eff = 1.0

        N_intervals, step_norm_mod, h_mod = self._choose_intervals_and_hmod(layer_height, g_eff, dr_clip)
        total_slices = N_intervals + 1  # including endpoints 0 and 1
        pbar = self._init_progress_bar(total_slices, layer_height, h_mod, progress_bar_width)

        slices: List[IsoSlice] = []
        skipped_count = 0

        cur = 0.0
        s0, npts, len_sum, wgrad_sum, curve_len = self._build_slice_at_level(
            cur, r, gn_face,
            orientation_hint=orientation_hint,
            deg=degree, samples=samples, periodic=periodic,
            dedupe_decimals=dedupe_decimals, edge_tol=edge_tol, verbose=verbose,
            min_component_points=min_component_points,
        )
        if s0 is not None:
            slices.append(s0)
        else:
            skipped_count += 1
        # update postfix first, then redraw with update()
        pbar.set_postfix(lvl=f"{cur:.6g}", length=f"{curve_len:8.2f}", refresh=False)
        pbar.update(1)

        intervals_done = 0
        while intervals_done < (N_intervals - 1) and (len(slices) + skipped_count) < max_levels:
            if len_sum > 0 and wgrad_sum > 0:
                avg_grad = wgrad_sum / len_sum
            else:
                avg_grad = g_eff

            delta_r_mod = h_mod * avg_grad
            rem_norm = max(0.0, 1.0 - cur)
            rem_intervals = N_intervals - intervals_done
            delta_r_goal = rem_norm / rem_intervals

            alpha = float(np.clip(controller_blend, 0.0, 1.0))
            delta_r = (1.0 - alpha) * delta_r_goal + alpha * delta_r_mod
            delta_r = float(np.clip(delta_r, 1e-4, 0.2))
            delta_r = min(delta_r, rem_norm)

            cur = cur + delta_r
            s_i, npts, len_sum, wgrad_sum, curve_len = self._build_slice_at_level(
                cur, r, gn_face,
                orientation_hint=orientation_hint,
                deg=degree, samples=samples, periodic=periodic,
                dedupe_decimals=dedupe_decimals, edge_tol=edge_tol, verbose=verbose,
                min_component_points=min_component_points,
            )
            if s_i is not None:
                slices.append(s_i)
            else:
                skipped_count += 1

            intervals_done += 1
            # update postfix first, then redraw with update()
            pbar.set_postfix(lvl=f"{cur:.6g}", length=f"{curve_len:8.2f}", refresh=False)
            pbar.update(1)

        if include_end and (cur < 1.0 - 1e-12 or (len(slices) == 0 or slices[-1].level < 1.0 - 1e-9)):
            cur = 1.0
            s_end, npts, len_sum, wgrad_sum, curve_len = self._build_slice_at_level(
                cur, r, gn_face,
                orientation_hint=orientation_hint,
                deg=degree, samples=samples, periodic=periodic,
                dedupe_decimals=dedupe_decimals, edge_tol=edge_tol, verbose=verbose,
                min_component_points=min_component_points,
            )
            if s_end is not None:
                slices.append(s_end)
            else:
                skipped_count += 1
            # ensure bar shows the final level/length before completion
            pbar.set_postfix(lvl=f"{cur:.6g}", length=f"{curve_len:8.2f}", refresh=False)
            if pbar.n < pbar.total:
                pbar.update(pbar.total - pbar.n)

        pbar.close()

        print(f"[SlicingBaseGraph] Adaptive extraction complete: {len(slices)} valid slices, {skipped_count} skipped")
        print("  Field normalized to [0, 1]")
        print(f"  Layer height (input)    : {layer_height:.6g} mm")
        print(f"  Layer height (modified) : {h_mod:.6g} mm  (N intervals = {N_intervals}, total slices = {total_slices})")

        return IsoSliceCollection(slices, show_progress=self.show_progress)

