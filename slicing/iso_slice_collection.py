"""Utilities for managing ordered collections of :class:`~slicing.iso_slice.IsoSlice`.

Maintainer: Abdallah Kamhawi (PhD researcher, DART Laboratory; Kamhawi@umich.edu)
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from tqdm import tqdm

from slicing.iso_slice import IsoSlice


class IsoSliceCollection:

    """Ordered container for iso-slices extracted at successive iso-levels."""

    def __init__(self, slices: Sequence[IsoSlice], *, show_progress: bool = True) -> None:
        """Initialise the collection and annotate each slice with its layer index.

        Parameters
        ----------
        slices : Sequence[IsoSlice]
            Iso-slices ordered by ascending iso-value.
        show_progress : bool, optional
            Display a progress bar while assigning layer numbers.
        """
        self.slices: List[IsoSlice] = list(slices)
        self.show_progress = bool(show_progress)

        enumerator: Iterable[int] = range(len(self.slices))
        iterator = tqdm(
            enumerator,
            desc="IsoSliceCollection assigning layer numbers",
            unit="slice",
            disable=not self.show_progress,
        )
        for idx in iterator:
            self.slices[idx].update_layer_number(idx)

    def __len__(self) -> int:
        """Return the number of stored slices."""
        return len(self.slices)

    def __getitem__(self, idx: int) -> IsoSlice:
        """Expose direct index-based access to individual slices."""
        return self.slices[idx]

    def levels(self) -> List[float]:
        """Return the normalised iso-values associated with every slice.

        Returns
        -------
        List[float]
            Iso-values sorted in the same order as ``self.slices``.
        """
        return [slice_.level for slice_ in self.slices]

    def total_length(self) -> float:
        """Sum the total polyline length across every slice component.

        Returns
        -------
        float
            Aggregated length measured in the same units as the source mesh.
        """
        return float(sum(slice_.total_length for slice_ in self.slices))

    def component_counts(self) -> List[int]:
        """Return the number of connected components per slice.

        Returns
        -------
        List[int]
            Component counts aligned with the order of ``self.slices``.
        """
        return [slice_.component_count for slice_ in self.slices]

