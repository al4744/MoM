from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.compile.config import load_workstream_c_config
from src.compile.runtime import configure_compile_environment


def test_defaults_disabled() -> None:
    cfg = load_workstream_c_config({})
    assert cfg.compile.enabled is False
    assert cfg.profile.enabled is False
    assert cfg.compile.targets == ("prefill", "decode")


def test_top_level_compile_config_parses() -> None:
    cfg = load_workstream_c_config(
        {
            "compile": {
                "enabled": True,
                "targets": ["decode"],
                "backend": "inductor",
                "fullgraph": True,
                "warmup_iters": 3,
            }
        }
    )
    assert cfg.compile.enabled is True
    assert cfg.compile.targets == ("decode",)
    assert cfg.compile.fullgraph is True
    assert cfg.compile.warmup_iters == 3


def test_legacy_engine_torch_compile_config_parses() -> None:
    cfg = load_workstream_c_config({"engine": {"torch_compile": {"enabled": True}}})
    assert cfg.compile.enabled is True


def test_invalid_compile_target_rejected() -> None:
    with pytest.raises(ValueError):
        load_workstream_c_config({"compile": {"targets": ["prefill", "bad"]}})


def test_compile_environment_sets_vllm_vars(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_TORCH_COMPILE_LEVEL", raising=False)
    cfg = load_workstream_c_config(
        {"compile": {"enabled": True, "fullgraph": False}}
    ).compile
    updates = configure_compile_environment(cfg)
    assert updates["VLLM_TORCH_COMPILE_LEVEL"] == "1"
    assert os.environ["VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE"] == "0"


def test_build_real_engine_sets_compile_env(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_TORCH_COMPILE_LEVEL", raising=False)
    captured: dict = {}

    class _FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_plugins = MagicMock()
    fake_plugins.set_torch_compile_backend = MagicMock()
    fake_vllm = MagicMock(LLM=_FakeLLM)

    with patch.dict(
        sys.modules,
        {"vllm": fake_vllm, "vllm.plugins": fake_plugins},
    ):
        from evaluation.engine_adapter import build_real_engine

        build_real_engine(
            {
                "model": {"name": "mock", "dtype": "float16"},
                "engine": {"retention": {"enabled": False}},
                "compile": {"enabled": True, "backend": "inductor"},
            }
        )

    assert captured["model"] == "mock"
    assert os.environ["VLLM_TORCH_COMPILE_LEVEL"] == "1"
    fake_plugins.set_torch_compile_backend.assert_called_once_with("inductor")
