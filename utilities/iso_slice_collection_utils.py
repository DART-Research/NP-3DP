"""Summary helpers for :class:`~slicing.iso_slice_collection.IsoSliceCollection`.

Maintainer: Abdallah Kamhawi (PhD researcher, DART Laboratory; Kamhawi@umich.edu)
"""

from __future__ import annotations

from typing import Dict

from slicing.iso_slice_collection import IsoSliceCollection




def summarise_components(collection: IsoSliceCollection) -> Dict[str, object]:



    """Derive high-level statistics about the connectivity of iso-slices.

    Parameters
    ----------
    collection : IsoSliceCollection
        Ordered collection returned by :meth:`slicing.slicing_base.SlicingBaseGraph.extract_iso_slices`.

    Returns
    -------
    Dict[str, object]
        Dictionary containing the following keys:

        ``total_slices``
            Total number of entries inside ``collection``.
        ``bifurcation_slices``
            How many slices exhibit more than one connected component.
        ``max_components``
            Maximum component count observed in a single slice.
        ``bifurcation_levels``
            List of iso-levels at which bifurcations occurred.

    Notes
    -----
    The counts are computed in a single pass over the stored slices. The
    method does not attempt to filter invalid components; it assumes that
    the collection has already been cleaned by the slicing pipeline.
    """

    stats: Dict[str, object] = {
        "total_slices": len(collection),
        "bifurcation_slices": 0,
        "max_components": 1,
        "bifurcation_levels": [],
    }

    for slice_obj in collection.slices:
        comp_count = getattr(slice_obj, "component_count", 1)
        if comp_count > 1:
            stats["bifurcation_slices"] += 1
            stats["bifurcation_levels"].append(slice_obj.level)
        if comp_count > stats["max_components"]:
            stats["max_components"] = comp_count
    return stats


