from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from src.retention.ttl_predictor import TTLPredictor

if TYPE_CHECKING:
    from vllm.sequence import SequenceGroup


@dataclass
class PinnedEntry:
    seq_group: Any          # SequenceGroup; held so blocks stay refcounted-alive
    expiry_time: float      # time.monotonic() deadline
    tool_name: str
    finish_time: float      # when this turn finished — for EMA update on reuse
    num_blocks: int         # GPU blocks held (for budget accounting)
    program_id: str


class PinManager:
    """Tracks SequenceGroups whose KV blocks should be retained post-finish.

    Pinning = deferred free.  Instead of calling block_manager.free() on a
    finished SequenceGroup, the scheduler hands it here.  The held reference
    keeps block refcounts > 0 so the KV stays on GPU.  When the next turn of
    the same agent program arrives, the scheduler calls free() on the held
    seq_group *before* allocating the new one; the prefix-cache hash lookup
    inside the allocator then finds the now-evictor-held blocks and revives
    them for the new sequence at zero reprefill cost.

    Thread safety
    -------------
    All methods must be called from the vLLM engine thread.  In debug builds
    (RETENTION_CHECK_THREAD=1 env var) an assertion enforces this.
    """

    _CHECK_THREAD = False  # set to True in tests that want the guard

    def __init__(
        self,
        predictor: TTLPredictor,
        total_gpu_blocks: int,
        max_pinned_fraction: float = 0.3,
    ) -> None:
        if total_gpu_blocks <= 0:
            raise ValueError("total_gpu_blocks must be positive")
        if not (0.0 < max_pinned_fraction <= 1.0):
            raise ValueError("max_pinned_fraction must be in (0, 1]")

        self.predictor = predictor
        self.total_gpu_blocks = total_gpu_blocks
        self.max_pinned_fraction = max_pinned_fraction
        self._max_pinned_blocks = int(total_gpu_blocks * max_pinned_fraction)

        # program_id → PinnedEntry
        self._pinned: dict[str, PinnedEntry] = {}
        self._pinned_blocks_total: int = 0

        self._owner_thread_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API (called from scheduler hooks A/B/C/D)
    # ------------------------------------------------------------------

    def pin(
        self,
        program_id: str,
        seq_group: Any,
        tool_name: str,
        num_blocks: int,
    ) -> bool:
        """Pin seq_group so its KV blocks are not freed yet.

        Returns False if the budget would be exceeded; caller must free normally.
        An existing pin for the same program_id is replaced (prev entry dropped
        — the scheduler will have already freed the old seq_group via reuse or
        TTL expiry before re-pinning).
        """
        self._check_thread()

        # Drop any stale entry for this program (shouldn't normally happen,
        # but guard against double-pin bugs).
        if program_id in self._pinned:
            old = self._pinned.pop(program_id)
            self._pinned_blocks_total -= old.num_blocks

        if self._pinned_blocks_total + num_blocks > self._max_pinned_blocks:
            return False

        tau = self.predictor.predict_ttl(tool_name)
        entry = PinnedEntry(
            seq_group=seq_group,
            expiry_time=time.monotonic() + tau,
            tool_name=tool_name,
            finish_time=time.monotonic(),
            num_blocks=num_blocks,
            program_id=program_id,
        )
        self._pinned[program_id] = entry
        self._pinned_blocks_total += num_blocks
        return True

    def try_reuse(self, program_id: str) -> Optional[PinnedEntry]:
        """Pop and return the pinned entry for program_id, or None on miss.

        Caller is responsible for:
          1. Calling scheduler._free_finished_seqs(entry.seq_group) to release
             the deferred hold (blocks flow into the evictor with hashes intact).
          2. Calling predictor.update(entry.tool_name, observed_duration) to
             close the EMA learning loop.
        """
        self._check_thread()
        entry = self._pinned.pop(program_id, None)
        if entry is None:
            return None
        self._pinned_blocks_total -= entry.num_blocks
        return entry

    def sweep_expired(self, waiting_program_ids: set[str]) -> list[PinnedEntry]:
        """Return all entries whose TTL has expired and whose program is not
        in the waiting queue (race-protection: don't expire an entry the
        moment before add_seq_group picks it up as a reuse).

        Caller must free the seq_group for each returned entry.
        """
        self._check_thread()
        now = time.monotonic()
        expired = []
        for pid, entry in list(self._pinned.items()):
            if now >= entry.expiry_time and pid not in waiting_program_ids:
                self._pinned.pop(pid)
                self._pinned_blocks_total -= entry.num_blocks
                expired.append(entry)
        return expired

    def evict_soonest_expiring(self) -> Optional[PinnedEntry]:
        """Pop and return the entry with the nearest expiry_time.

        Used as a pressure valve when the scheduler cannot allocate a new
        request.  Caller must free the seq_group.
        """
        self._check_thread()
        if not self._pinned:
            return None
        pid = min(self._pinned, key=lambda p: self._pinned[p].expiry_time)
        entry = self._pinned.pop(pid)
        self._pinned_blocks_total -= entry.num_blocks
        return entry

    def num_pinned_entries(self) -> int:
        return len(self._pinned)

    def num_pinned_blocks(self) -> int:
        return self._pinned_blocks_total

    # ------------------------------------------------------------------
    # Thread-safety guard (debug only)
    # ------------------------------------------------------------------

    def bind_to_current_thread(self) -> None:
        """Call once from the engine thread after construction."""
        self._owner_thread_id = threading.get_ident()

    def _check_thread(self) -> None:
        if not self._CHECK_THREAD:
            return
        current = threading.get_ident()
        assert self._owner_thread_id is None or current == self._owner_thread_id, (
            f"PinManager called from thread {current}, "
            f"expected {self._owner_thread_id} (engine thread)"
        )
