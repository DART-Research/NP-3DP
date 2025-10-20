"""Shared Plotly helpers resilient to transient browser disconnects."""

from __future__ import annotations

import errno
from typing import Callable


_ECONNRESET_CODES = {errno.ECONNRESET}
if hasattr(errno, "WSAECONNRESET"):
    _ECONNRESET_CODES.add(getattr(errno, "WSAECONNRESET"))


def guard_connection_reset(action: Callable[[], None], *, context: str) -> None:
    """Execute `action` while swallowing connection resets from Plotly's server."""
    try:
        action()
    except OSError as err:
        err_no = getattr(err, "errno", None)
        win_err = getattr(err, "winerror", None)
        if isinstance(err, ConnectionResetError) or err_no in _ECONNRESET_CODES or win_err in _ECONNRESET_CODES:
            print(f"[viz] Ignore connection reset during {context}: {err}")
            return
        raise
