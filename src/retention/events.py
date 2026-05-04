from __future__ import annotations

import json
import sys
import time
from typing import IO, Optional

_log_file: Optional[IO[str]] = None
_wandb = None


def configure(log_file: Optional[str] = None, use_wandb: bool = True) -> None:
    """Configure event output. Call once at engine startup.

    Args:
        log_file: Path to append JSON-lines to. None → stderr.
        use_wandb: Attempt to import wandb and log there too.
    """
    global _log_file, _wandb

    if log_file is not None:
        _log_file = open(log_file, "a")  # noqa: WPS515

    if use_wandb:
        try:
            import wandb as _w
            _wandb = _w
        except ImportError:
            pass


def log_event(
    event_type: str,
    program_id: str,
    *,
    tool_name: Optional[str] = None,
    num_blocks: int = 0,
    ttl_assigned: Optional[float] = None,
    observed_duration: Optional[float] = None,
    current_pinned_count: int = 0,
    current_pinned_blocks: int = 0,
) -> None:
    """Emit one structured retention event.

    event_type values: pin, reuse, expire, evict, pin_rejected_budget
    """
    record = {
        "event_type": event_type,
        "monotonic_ts": time.monotonic(),
        "wallclock_ts": time.time(),
        "program_id": program_id,
        "tool_name": tool_name,
        "num_blocks": num_blocks,
        "ttl_assigned": ttl_assigned,
        "observed_duration": observed_duration,
        "current_pinned_count": current_pinned_count,
        "current_pinned_blocks": current_pinned_blocks,
    }
    line = json.dumps(record)

    if _log_file is not None:
        _log_file.write(line + "\n")
        _log_file.flush()
    else:
        print(line, file=sys.stderr)

    if _wandb is not None:
        try:
            _wandb.log(record)
        except Exception:
            pass
