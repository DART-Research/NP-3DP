# NP Slicing Toolkit

NP Slicing is a graph-based pipeline for extracting topology-aware iso-curves from triangulated surfaces. It builds a weighted vertex graph from the mesh, detects and prunes critical points, synthesizes a boundary-conforming scalar field, and adaptively samples iso-curves with multi-component support. This document expands the high-level overview into a step-by-step, reproducible description with equations and algorithmic details suitable as a base for a scientific report.

## Citation
If you use this code or algorithm in your research, please cite: Non-Planar Slicing for High-Genus Surfaces with Non-Coplanar Interfaces

**Point of Contact**
- Maintainer: Abdallah Kamhawi (PhD researcher, DART Laboratory; Kamhawi@umich.edu)

**Entrypoint**
- Orchestrating script: `main.py:33`

**Core Modules**
- Graph + slicing: `slicing/slicing_base.py:33`
- Critical points: `slicing/critical_points.py:753`
- Iso-curve container: `slicing/iso_slice.py:21`
- Multi-component assembly: `slicing/multi_component_slicing.py:44`
- Mesh loader: `loaders/mesh_loader.py:15`

**Defaults**
- Slicing defaults: `slicing/config.py`
- Visualization defaults: `viz/config.py`

**Typical Outputs**
- Scalar field visual: `scalar_field.html`
- Slicing graph: `slicing_graph.html`
- Iso-curves: `iso_curves.html`
- Subset fields: `scalar_upper.html`, `scalar_lower.html`, `scalar_saddles.html`

**Dependencies**
- Python ≥ 3.8; NumPy, SciPy, NetworkX, tqdm, trimesh, plotly, matplotlib; numba (optional JIT).

**Run**
- `python main.py`

The demo loads an example OBJ, builds the slicing graph, detects saddles, constructs a conforming scalar field, extracts iso-curves, prints summary stats, and emits Plotly HTML reports.

**Abstract**
- We construct a scalar field F: V → [0, 1] on the mesh vertices that adheres to boundary conditions and respects saddle constraints. F combines geodesic distances to upper/lower boundary sets with saddle-biased offsets. Iso-curves of F are sampled with an adaptive controller driven by local gradient magnitudes to approximate a requested physical layer height.

**Notation**
- V: set of mesh vertices, F faces, E edges.
- G = (V, E): undirected vertex graph with edge weight w(u, v) = ||p_u − p_v||_2.
- B_upper, B_lower: boundary vertex sets (geodesic boundary loops classified by axis).
- S: set of saddle vertices.
- D_x→Y: geodesic distance from x to the set Y (multi-source Dijkstra on G).

**Step‑By‑Step Pipeline**
- Mesh Loading
  - `MeshLoader` wraps `trimesh.Trimesh`, preserves materials, and exposes simple Plotly helpers (`loaders/mesh_loader.py:15`).
- Graph Construction
  - `SlicingBaseGraph` builds G with Euclidean edge weights and detects boundary loops from edges with single face incidence; loops are stored on nodes/edges as attributes (`slicing/slicing_base.py:33`). CSR arrays are prepared for fast JIT Dijkstra.
- Boundary Classification
  - Loops are classified as lower/upper along a chosen axis with tolerance; middle loops optionally assigned to nearest extreme (`label_lower_upper_boundaries`, `slicing/slicing_base.py:618`). Interior vertices inherit the closest loop on each side via multi-source geodesics (`assign_nearest_lower_upper_boundaries`, `slicing/slicing_base.py:707`).
- Critical Point Detection
  - Piecewise-linear link test over each vertex’s one-ring counts sign changes of f(v)−f(u_i) (ordered ring). Classification: C=0 with all positive→maximum, all negative→minimum; C=2→regular; C≥4 and even→saddle (optionally allow certain odd counts). Optional Laplacian smoothing, adaptive epsilon per vertex, and confidence scoring (`slicing/critical_points.py:753`). Nearby saddles can be clustered by geodesic distance with a cutoff and ranked by confidence or local saliency.
- Conforming Scalar Field F
  - For each saddle s∈S estimate a radius R_s (min distance from s to either boundary set, unless overridden). For all vertices v, compute distances D_vs from v to each s, and D_sn to boundaries from saddles. Form weights:
    - X_vs = sqrt(max(D_vs + R_s^n − R_s, 0))
    - W_vs ∝ 1 / sinh(max(X_vs, 1e−12)), row-normalized across s
  - Vertex-to-boundary distances D_vn are biased by B = W·D_sn, producing T_up = D_vn_up + B_up and T_lo = D_vn_lo + B_lo. Reduce to scalar distances R_upper = min_n T_up[:, n], R_lower = min_n T_lo[:, n]. Blend to a unit range field:
    - F(v) = R_upper(v) / (R_upper(v) + R_lower(v) + ε)
  - Implementation: `compute_conforming_scalar_field`, `slicing/slicing_base.py:772`.
- Iso-Curve Extraction
  - Normalize field r = (f − min)/(max − min). Estimate per-face |∇r| via a local 2D parameterization of each triangle; gradient magnitude is sqrt(a^2 + b^2) with a = (r_j − r_i)/x_j and b = (r_k − r_i − x_k·a)/y_k (`_compute_face_grad_norms`, `slicing/slicing_base.py:958`).
  - For each iso-level ℓ, intersect r with triangle edges; keep edges where r crosses ℓ within tolerance; compute segment endpoints by linear interpolation; accumulate segment length and gradient-weighted length (`_build_slice_at_level`, `slicing/slicing_base.py:1012`).
  - Group raw segments into connected components by KD-tree proximity and exact endpoint hashing; order segments into polylines; close loops when degrees are balanced; filter tiny components (`build_slice_components`, `slicing/multi_component_slicing.py:44`).
  - For each component, sample a representative polyline by greedy nearest-neighbour ordering, uniform arc-length reparameterization, and Savitzky–Golay smoothing; compute per-component and total lengths (`slicing/iso_slice.py:21`).
- Adaptive Controller (Layer Height)
  - Let g_eff be the median of finite |∇r| over faces. Target normalized step ∆r_target = clip(h_in·g_eff, dr_clip). Choose N ≈ round(1/∆r_target); set h_mod = (1/N)/g_eff (`_choose_intervals_and_hmod`, `slicing/slicing_base.py:1126`).
  - During integration, blend remaining-range spacing with gradient-driven spacing: ∆r = (1−α)·(remaining_norm/remaining_intervals) + α·(h_mod·avg_grad), clipped to dr_clip; avg_grad is accumulated from the last slice’s segments via gradient-weighted lengths. Ensure the terminal iso-level 1.0 is included when configured (`extract_iso_slices`, `slicing/slicing_base.py:1172`).

**Key Algorithms and Equations**
- Graph distances
  - Multi-source Dijkstra on G for geodesics to boundary sets and saddle sets. Complexity: O(E log V) with a binary heap; implemented in pure NetworkX and optionally in JIT over CSR (`slicing/numba_accel.py`).
- Critical points on PL manifolds
  - Upper/lower link sign-change test on one-ring neighbourhoods classifies minima/maxima/saddles; optional per-vertex adaptive ε based on local variation percentiles; optional Laplacian smoothing of the scalar field; optional geodesic clustering of saddle candidates with DSU union-find.
- Scalar field blend
  - F(v) = R_up(v) / (R_up(v) + R_lo(v) + ε), where R_up / R_lo are boundary distances biased by saddle-aware W. Weighting uses W_vs ∝ csch(X_vs) with X_vs defined above; exponent n controls falloff around saddles.
- Gradient estimation on triangles
  - Map triangle to a local (u, v) frame and solve affine coefficients a, b for r(u, v) = a·u + b·v + c; use |∇r| = sqrt(a^2 + b^2).
- Multi-component assembly
  - KD-tree radius = connectivity_factor × median segment length; exact endpoint hashing at configurable decimals fuses near-duplicates; components are ordered by greedy traversal over hashed adjacency.
- Adaptive spacing
  - Controller blends uniform coverage of [0,1] with local gradient pacing to approximate a physical layer height under the mesh’s scalar variation.

**Data Structures and Annotations**
- Graph nodes store `pos`, `node_type` ∈ {boundary, intra}, `boundary_number`, `boundary_side`, nearest-side ids and distances, `morse_index` ∈ {−1,0,1,2}, `critical_type`, optional `critical_confidence`, and `conforming_scalar`.
- Graph edges store `weight`, `edge_type` ∈ {internal, boundary}, `boundary_number`, and `boundary_side`.


**Pseudocode (High Level)**
- Load mesh → build G
- Label boundary loops and sides
- Compute height field along axis; detect saddles (optionally multi-scale and clipped)
- Build F with W-weights and biased boundary geodesics
- Normalize r, compute |∇r|, choose N and h_mod
- For each level: intersect triangles → segments → components → sample curves → store `IsoSlice`

**Complexity and Performance Notes**
- Dijkstra: O(E log V); JIT kernels reduce Python overhead and expose CSR to Numba for distance and extraction loops.
- Per-face gradients and iso-segment extraction run in O(F) per level (vectorized or JIT); adaptive controller keeps level count near 1/∆r_target.
- Memory: CSR graph stores O(E) indices/weights; intermediate D matrices are streamed per source to bound peak memory when possible.

**Limitations and Assumptions**
- Scalar field is piecewise-linear; iso-intersections are linear per face.
- Boundary detection expects well-formed manifold edges; non-manifold cases may degrade classification.
- Confidence and persistence are heuristic and serve ranking/filtering rather than exact topology guarantees.

**Module Map**
- `loaders` — mesh ingestion and Plotly helpers.
- `slicing` — graph construction, distances, critical points, scalar field, adaptive extraction, and JIT accelerators.
- `utilities` — mesh scale heuristics, configuration helpers, iso-slice statistics.
- `viz` — scalar/graph/slice Plotly visualizations with YAML-driven defaults.

**References (background)**
- Dijkstra, 1959. A note on two problems in connexion with graphs.
- Edelsbrunner et al. Morse theory on piecewise linear manifolds.
- Savitzky & Golay, 1964. Smoothing and differentiation of data by simplified least squares procedures.

