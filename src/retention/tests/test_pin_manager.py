"""Unit tests for PinManager.

seq_group is mocked with a plain object — PinManager stores it opaquely.
The block-count is passed explicitly by the caller (scheduler), so no vLLM
imports are needed here.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.retention.pin_manager import PinManager, PinnedEntry
from src.retention.ttl_predictor import TTLPredictor


def _predictor(ttl: float = 1.0) -> TTLPredictor:
    """Predictor that always returns a fixed TTL (alpha=1, safety=1)."""
    p = TTLPredictor(alpha=1.0, default_ttl=ttl, safety_factor=1.0)
    return p


def _seq_group(program_id: str = "prog-1") -> SimpleNamespace:
    return SimpleNamespace(program_id=program_id)


class TestPin:
    def test_pin_returns_true_when_budget_allows(self):
        pm = PinManager(_predictor(), total_gpu_blocks=100)
        sg = _seq_group()
        assert pm.pin("prog-1", sg, "pytest", num_blocks=10) is True

    def test_pin_rejects_when_over_budget(self):
        pm = PinManager(_predictor(), total_gpu_blocks=100,
                        max_pinned_fraction=0.1)  # max 10 blocks
        sg = _seq_group()
        assert pm.pin("prog-1", sg, "pytest", num_blocks=11) is False

    def test_pin_counts_toward_budget(self):
        pm = PinManager(_predictor(), total_gpu_blocks=100,
                        max_pinned_fraction=0.2)  # max 20 blocks
        pm.pin("prog-1", _seq_group("prog-1"), "pytest", num_blocks=15)
        # 15 used; 10 more would exceed 20
        assert pm.pin("prog-2", _seq_group("prog-2"), "pytest",
                      num_blocks=10) is False
        # 5 more fits
        assert pm.pin("prog-3", _seq_group("prog-3"), "pytest",
                      num_blocks=5) is True

    def test_num_pinned_entries_reflects_pins(self):
        pm = PinManager(_predictor(), total_gpu_blocks=100)
        assert pm.num_pinned_entries() == 0
        pm.pin("p1", _seq_group("p1"), "t", num_blocks=5)
        pm.pin("p2", _seq_group("p2"), "t", num_blocks=5)
        assert pm.num_pinned_entries() == 2

    def test_num_pinned_blocks_accumulates(self):
        pm = PinManager(_predictor(), total_gpu_blocks=100)
        pm.pin("p1", _seq_group("p1"), "t", num_blocks=7)
        pm.pin("p2", _seq_group("p2"), "t", num_blocks=3)
        assert pm.num_pinned_blocks() == 10

    def test_double_pin_same_program_replaces_entry(self):
        pm = PinManager(_predictor(), total_gpu_blocks=100)
        sg1 = _seq_group("p1")
        sg2 = _seq_group("p1")
        pm.pin("p1", sg1, "t", num_blocks=5)
        pm.pin("p1", sg2, "t", num_blocks=3)
        assert pm.num_pinned_entries() == 1
        assert pm.num_pinned_blocks() == 3  # old 5 was removed


class TestTryReuse:
    def test_hit_returns_entry_and_pops(self):
        pm = PinManager(_predictor(), total_gpu_blocks=100)
        sg = _seq_group()
        pm.pin("p1", sg, "pytest", num_blocks=8)
        entry = pm.try_reuse("p1")
        assert entry is not None
        assert entry.seq_group is sg
        assert entry.tool_name == "pytest"
        assert pm.num_pinned_entries() == 0
        assert pm.num_pinned_blocks() == 0

    def test_miss_returns_none(self):
        pm = PinManager(_predictor(), total_gpu_blocks=100)
        assert pm.try_reuse("nonexistent") is None

    def test_reuse_reduces_block_count(self):
        pm = PinManager(_predictor(), total_gpu_blocks=100)
        pm.pin("p1", _seq_group("p1"), "t", num_blocks=10)
        pm.pin("p2", _seq_group("p2"), "t", num_blocks=5)
        pm.try_reuse("p1")
        assert pm.num_pinned_blocks() == 5


class TestSweepExpired:
    def test_expired_entries_are_returned(self):
        pm = PinManager(_predictor(ttl=0.001), total_gpu_blocks=100)
        pm.pin("p1", _seq_group("p1"), "t", num_blocks=5)
        time.sleep(0.01)
        expired = pm.sweep_expired(waiting_program_ids=set())
        assert len(expired) == 1
        assert expired[0].program_id == "p1"
        assert pm.num_pinned_entries() == 0

    def test_non_expired_entries_are_kept(self):
        pm = PinManager(_predictor(ttl=9999), total_gpu_blocks=100)
        pm.pin("p1", _seq_group("p1"), "t", num_blocks=5)
        expired = pm.sweep_expired(waiting_program_ids=set())
        assert expired == []
        assert pm.num_pinned_entries() == 1

    def test_race_protection_keeps_entry_if_in_waiting(self):
        pm = PinManager(_predictor(ttl=0.001), total_gpu_blocks=100)
        pm.pin("p1", _seq_group("p1"), "t", num_blocks=5)
        time.sleep(0.01)
        # p1 is expired but also in the waiting queue — don't expire it
        expired = pm.sweep_expired(waiting_program_ids={"p1"})
        assert expired == []
        assert pm.num_pinned_entries() == 1

    def test_mixed_expiry(self):
        pm = PinManager(_predictor(ttl=9999), total_gpu_blocks=100)
        pm.pin("soon", _seq_group("soon"), "t", num_blocks=4)
        pm.pin("later", _seq_group("later"), "t", num_blocks=4)
        # Force expiry_time in the past for "soon"
        pm._pinned["soon"].expiry_time = time.monotonic() - 1.0
        expired = pm.sweep_expired(waiting_program_ids=set())
        assert len(expired) == 1
        assert expired[0].program_id == "soon"
        assert pm.num_pinned_entries() == 1


class TestEvictSoonestExpiring:
    def test_returns_none_when_empty(self):
        pm = PinManager(_predictor(), total_gpu_blocks=100)
        assert pm.evict_soonest_expiring() is None

    def test_returns_soonest_entry(self):
        pm = PinManager(_predictor(ttl=9999), total_gpu_blocks=100)
        pm.pin("p1", _seq_group("p1"), "t", num_blocks=5)
        pm.pin("p2", _seq_group("p2"), "t", num_blocks=5)
        # Make p2 expire sooner
        pm._pinned["p2"].expiry_time = time.monotonic() + 1.0
        pm._pinned["p1"].expiry_time = time.monotonic() + 100.0
        victim = pm.evict_soonest_expiring()
        assert victim is not None
        assert victim.program_id == "p2"
        assert pm.num_pinned_entries() == 1

    def test_reduces_block_count(self):
        pm = PinManager(_predictor(), total_gpu_blocks=100)
        pm.pin("p1", _seq_group("p1"), "t", num_blocks=12)
        pm.evict_soonest_expiring()
        assert pm.num_pinned_blocks() == 0


class TestBudgetValidation:
    def test_zero_total_blocks_raises(self):
        with pytest.raises(ValueError):
            PinManager(_predictor(), total_gpu_blocks=0)

    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError):
            PinManager(_predictor(), total_gpu_blocks=100,
                       max_pinned_fraction=0.0)
        with pytest.raises(ValueError):
            PinManager(_predictor(), total_gpu_blocks=100,
                       max_pinned_fraction=1.1)


class TestThreadGuard:
    def test_wrong_thread_raises_when_check_enabled(self):
        import threading

        pm = PinManager(_predictor(), total_gpu_blocks=100)
        pm._CHECK_THREAD = True
        pm.bind_to_current_thread()

        errors = []

        def _off_thread():
            try:
                pm.num_pinned_entries()  # doesn't check thread
                pm.pin("p", _seq_group(), "t", num_blocks=1)
                errors.append("expected AssertionError not raised")
            except AssertionError:
                pass  # expected
            except Exception as e:
                errors.append(str(e))

        t = threading.Thread(target=_off_thread)
        t.start()
        t.join()
        assert not errors


class TestPredictorIntegration:
    def test_ttl_is_applied_from_predictor(self):
        p = TTLPredictor(alpha=1.0, default_ttl=5.0, safety_factor=1.0)
        pm = PinManager(p, total_gpu_blocks=100)
        sg = _seq_group()
        pm.pin("p1", sg, "pytest", num_blocks=1)
        entry = pm._pinned["p1"]
        # expiry should be ~5 seconds from now
        remaining = entry.expiry_time - time.monotonic()
        assert 4.9 < remaining < 5.1

    def test_finish_time_is_set(self):
        pm = PinManager(_predictor(), total_gpu_blocks=100)
        before = time.monotonic()
        pm.pin("p1", _seq_group(), "t", num_blocks=1)
        after = time.monotonic()
        entry = pm._pinned["p1"]
        assert before <= entry.finish_time <= after
