"""Utilities for coercing configuration values into canonical types.

Maintainer: Abdallah Kamhawi (PhD researcher, DART Laboratory; Kamhawi@umich.edu)
"""

from __future__ import annotations

from typing import Iterable, Sequence


def coerce_axis(value: object, fallback: str) -> str:

    """Return a canonical axis label drawn from {'x', 'y', 'z'}.

    Parameters
    ----------
    value : object
        Free-form axis indicator supplied by configuration. Strings are
        normalised by trimming whitespace and lowering the case.
    fallback : str
        Axis label to fall back to when ``value`` cannot be mapped onto one of
        the recognised axis identifiers.

    Returns
    -------
    str
        Either the normalised axis string or ``fallback`` if the candidate was
        invalid.
    """
    axis = str(value or fallback).lower()
    if axis not in {"x", "y", "z"}:
        return fallback
    return axis


def coerce_float(value: object, fallback: float) -> float:
    """Cast ``value`` to ``float`` while guarding against config noise.

    Parameters
    ----------
    value : object
        Arbitrary configuration token that should represent a floating point
        number. Strings and numeric values are accepted.
    fallback : float
        Value returned when the cast fails.

    Returns
    -------
    float
        ``value`` converted to ``float`` or ``fallback`` if conversion fails.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def coerce_int(value: object, fallback: int) -> int:
    """Cast ``value`` to ``int`` while preserving a safe default.

    Parameters
    ----------
    value : object
        Candidate integer encoded as a number or string.
    fallback : int
        Value returned when ``value`` cannot be interpreted as an integer.

    Returns
    -------
    int
        ``value`` converted to ``int`` or ``fallback`` on error.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def coerce_bool(value: object, fallback: bool) -> bool:
    """Interpret ``value`` as a boolean using common textual conventions.

    Parameters
    ----------
    value : object
        Configuration token that should represent ``True`` or ``False``. Truthy
        and falsy strings, numbers, and actual booleans are recognised.
    fallback : bool
        Value returned when ``value`` cannot be interpreted reliably.

    Returns
    -------
    bool
        Parsed boolean or ``fallback`` if the conversion is ambiguous.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return fallback


def coerce_sequence(values: object) -> Iterable[float]:
    """Coerce configuration input into a homogeneous sequence of floats.

    Parameters
    ----------
    values : object
        Either a scalar or a sequence of scalars. Strings are treated as
        scalars rather than iterables to avoid splitting digits.

    Returns
    -------
    Iterable[float]
        A list of floats converted elementwise via :func:`coerce_float`.
        Returns an empty list when ``values`` is ``None``.
    """
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return [coerce_float(v, 0.0) for v in values]
    if values is not None:
        return [coerce_float(values, 0.0)]
    return []

