"""Immutable-style container for iso-curve geometry and metadata.

The class records the raw intersection points, one or more connected curve
components, and a smoothed representative polyline sampled at a user-defined
resolution. The helper methods encapsulate the ordering, filtering, and
smoothing steps required by the slicing pipeline while keeping the underlying
geometry accessible for further analysis.

Maintainer: Abdallah Kamhawi (PhD researcher, DART Laboratory; Kamhawi@umich.edu)
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm


class IsoSlice:
    """Represent a single iso-level with all of its connected curve components."""

    def __init__(
        self,
        level: float,
        points: Sequence[Tuple[float, float, float]] | np.ndarray,
        *,
        degree: int = 3,
        samples: int = 5,
        periodic: bool = True,
        components: Sequence[np.ndarray] | None = None,
        component_lengths: Sequence[float] | None = None,
        points_ordered: bool = False,
        show_progress: bool = True,
    ) -> None:
        """Normalise raw intersection data and prepare derived geometry.

        Parameters
        ----------
        level : float
            Normalised iso-value assigned to the slice.
        points : Sequence[Tuple[float, float, float]] | np.ndarray
            Raw intersection points for the primary component. If empty, the
            first component from ``components`` becomes the primary curve.
        degree : int, optional
            Retained for metadata completeness. The current implementation
            produces a piecewise-linear approximation rather than a B-spline.
        samples : int, optional
            Number of samples requested for the representative curve.
        periodic : bool, optional
            Whether the primary curve should be closed by repeating the first
            point at the end of the sequence.
        components : Sequence[np.ndarray], optional
            Additional connected components forming the slice. Each array must
            be shaped ``(N_i, 3)``.
        component_lengths : Sequence[float], optional
            Precomputed arc lengths for ``components``. When omitted the lengths
            are recomputed from the provided coordinates.
        points_ordered : bool, optional
            Set to ``True`` when ``points`` already follow the desired traversal
            order, disabling the internal nearest-neighbour reordering step.
        show_progress : bool, optional
            Toggle progress bars for the ordering, smoothing, and sampling stages.

        Notes
        -----
        Sampling proceeds by ordering the raw points with a greedy
        nearest-neighbour heuristic, measuring cumulative arc length, and then
        linearly interpolating onto a uniformly spaced parameter grid. The
        Savitzky-Golay smoother operates independently on each coordinate axis
        to suppress high-frequency jitter introduced during extraction.
        """
        self.level = float(level)
        self.layer_number = 0
        self.degree = int(degree)
        self.samples = int(samples)
        self.periodic = bool(periodic)
        self.show_progress = bool(show_progress)
        self.points_ordered = bool(points_ordered)

        primary_points = np.asarray(points, dtype=float)
        if primary_points.size == 0 and components:
            primary_points = np.asarray(components[0], dtype=float)
        if primary_points.size == 0:
            primary_points = np.empty((0, 3), dtype=float)
        else:
            primary_points = primary_points.reshape((-1, 3))

        self.points = [tuple(map(float, pt)) for pt in primary_points]

        if components is not None:
            converted = [np.asarray(comp, dtype=float).reshape((-1, 3)) for comp in components]
        else:
            converted = [primary_points] if primary_points.size else []
        self.components = converted

        if component_lengths is not None:
            self.component_lengths = [float(length) for length in component_lengths]
        else:
            self.component_lengths = [
                self._compute_polyline_length(comp)
                for comp in self.components
            ]

        self.curve = self._make_spline(
            self.points,
            self.degree,
            self.samples,
            assume_ordered=self.points_ordered,
        )
        self.curve_length = self._compute_polyline_length(self.curve)
        self.total_length = (
            float(sum(self.component_lengths))
            if self.component_lengths
            else self.curve_length
        )
        self.component_count = len(self.components)

    def update_layer_number(self, layer_number: int) -> None:
        """Record the zero-based layer index assigned by the collection.

        Parameters
        ----------
        layer_number : int
            Sequential index injected by :class:`slicing.iso_slice_collection.IsoSliceCollection`.
        """
        self.layer_number = int(layer_number)

    def _order_nearest(self, arr: np.ndarray) -> np.ndarray:
        """Order raw points via a greedy nearest-neighbour tour.

        Parameters
        ----------
        arr : np.ndarray
            Set of unordered intersection points shaped ``(N, 3)``.

        Returns
        -------
        np.ndarray
            Points reordered to minimise the sum of Euclidean jumps produced
            by the greedy traversal.
        """
        n = arr.shape[0]
        if n < 2:
            return arr.copy()

        visited = np.zeros(n, dtype=bool)
        start_idx = int(np.argmin(arr[:, 0]))
        order = [start_idx]
        visited[start_idx] = True

        progress_iter: Iterable[int] = range(n - 1)
        iterator = tqdm(
            progress_iter,
            desc=f"IsoSlice[level={self.level:.4f}] ordering",
            unit="step",
            leave=False,
            disable=not self.show_progress,
        )

        for _ in iterator:
            last_point = arr[order[-1]]
            remaining = np.where(~visited)[0]
            next_idx = int(
                np.argmin(
                    np.linalg.norm(arr[remaining] - last_point, axis=1)
                )
            )
            nxt = remaining[next_idx]
            order.append(nxt)
            visited[nxt] = True

        return arr[order]

    def _clean_and_smooth(
        self,
        path: np.ndarray,
        *,
        min_dist: float = 1e-6,
        smooth_window: int = 7,
        smooth_poly: int = 2,
    ) -> np.ndarray:
        """Cull near-duplicate vertices and optionally smooth the curve.

        Parameters
        ----------
        path : np.ndarray
            Polyline shaped ``(N, 3)`` produced by :meth:`_order_nearest`.
        min_dist : float, optional
            Minimum edge length tolerated before vertices are merged.
        smooth_window : int, optional
            Window length for the Savitzky-Golay filter (forced to an odd
            value not exceeding ``N``).
        smooth_poly : int, optional
            Polynomial degree used by the Savitzky-Golay filter.

        Returns
        -------
        np.ndarray
            Filtered polyline.
        """
        if path.shape[0] < 2:
            return path

        diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
        keep = np.concatenate([[True], diffs > min_dist])
        filtered = path[keep]

        if filtered.shape[0] < smooth_window:
            return filtered

        window = min(smooth_window, (filtered.shape[0] // 2) * 2 + 1)
        iterator: Iterable[int] = range(3)
        for axis in tqdm(
            iterator,
            desc=f"IsoSlice[level={self.level:.4f}] smoothing",
            unit="axis",
            leave=False,
            disable=not self.show_progress,
        ):
            filtered[:, axis] = savgol_filter(
                filtered[:, axis],
                window_length=window,
                polyorder=smooth_poly,
                mode="mirror",
            )
        return filtered

    def _make_spline(
        self,
        pts: Sequence[Tuple[float, float, float]] | np.ndarray,
        degree: int,
        samples: int,
        *,
        assume_ordered: bool = False,
    ) -> np.ndarray:
        """Construct a uniformly sampled polyline approximation of the curve.

        Parameters
        ----------
        pts : Sequence
            Collection of seed points describing the curve.
        degree : int
            Stored for compatibility with downstream consumers; not used in the
            current linear sampling routine.
        samples : int
            Number of points to evaluate along the curve, including the
            endpoints.
        assume_ordered : bool, optional
            When ``True`` the input point sequence is treated as already ordered
            and the nearest-neighbour traversal is skipped.

        Returns
        -------
        np.ndarray
            Array of sampled points shaped ``(samples, 3)`` (or fewer when the
            input was degenerate).
        """
        coords = np.asarray(list(pts), dtype=float)
        if coords.shape[0] < 2:
            return coords

        if assume_ordered:
            ordered = coords.copy()
        else:
            ordered = self._order_nearest(coords)

        if self.periodic:
            if ordered.shape[0] < 2:
                return ordered
            if not np.allclose(ordered[0], ordered[-1]):
                ordered = np.vstack([ordered, ordered[0]])
        elif ordered.shape[0] > 1 and np.allclose(ordered[0], ordered[-1]):
            ordered = ordered[:-1]

        segments = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
        distances = np.concatenate([[0.0], np.cumsum(segments)])
        length = float(distances[-1])
        if length < 1e-12:
            return ordered[:-1] if self.periodic else ordered

        sample_distances = np.linspace(0.0, length, max(samples, 2))
        sampled: List[np.ndarray] = []
        iterator = tqdm(
            sample_distances,
            desc=f"IsoSlice[level={self.level:.4f}] sampling",
            leave=False,
            disable=not self.show_progress,
        )
        for target in iterator:
            idx = int(np.searchsorted(distances, target))
            if idx == 0:
                sampled.append(ordered[0])
                continue
            seg_len = segments[idx - 1]
            if seg_len < 1e-12:
                sampled.append(ordered[idx])
                continue
            ratio = (target - distances[idx - 1]) / seg_len
            interpolated = ordered[idx - 1] + ratio * (ordered[idx] - ordered[idx - 1])
            sampled.append(interpolated)

        spline = self._clean_and_smooth(np.array(sampled, dtype=float))
        if self.periodic and spline.shape[0] > 0:
            mask = np.concatenate([[True], np.any(np.diff(spline, axis=0), axis=1)])
            spline = spline[mask]
            spline = np.vstack([spline, spline[0]])
        return spline

    @staticmethod
    def _compute_polyline_length(path: np.ndarray) -> float:
        """Return the cumulative Euclidean arc length of ``path``."""
        if path.shape[0] < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))

    def get_curve_points(self) -> np.ndarray:
        """Return the sampled primary curve as a defensive copy."""
        return self.curve.copy()

    def get_original_points(self) -> np.ndarray:
        """Return the original intersection points as a NumPy array."""
        return np.array(self.points, dtype=float)

    def get_component_curves(self) -> List[np.ndarray]:
        """Return copies of every connected component stored on the slice."""
        return [comp.copy() for comp in self.components]

