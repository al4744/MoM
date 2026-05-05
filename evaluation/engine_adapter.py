"""Workstream D — Engine adapter layer.

Three things:

  1. ``EngineProtocol`` — the minimal vLLM-shape interface ``run_real()`` needs.
     Defining it explicitly lets D run end-to-end without importing vLLM,
     which means MockEngine works on CPU for smoke tests + CI.

  2. ``MockEngine`` — deterministic fake that simulates ``LLM.generate()``.
     Honours sampling_params.max_tokens, fakes per-token timing, returns
     RequestOutput-shaped objects with ``.metrics`` and ``.outputs[0]``.
     Used by the smoke test and for any pre-GPU local development.

  3. ``build_real_engine(cfg)`` — lazy factory that imports vLLM and constructs
     an ``LLM`` from a parsed config dict. Returns the engine; caller drives
     the loop. Raises a clear error if vLLM is not installed (so dry-run +
     mock paths don't pay an import cost).

The retention/quantization/torch.compile knobs are forwarded where supported
by Daksh's vLLM patches and otherwise marked TODO with the upstream gap.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from vllm import SamplingParams  # noqa: F401


# ---------------------------------------------------------------------------
# Protocol — the surface run_real() depends on
# ---------------------------------------------------------------------------

class EngineProtocol(Protocol):
    """Minimal duck-typed interface required by ``run_eval.run_trace``.

    Both the real ``vllm.LLM`` and ``MockEngine`` below satisfy this. We do
    not require any other vLLM symbols at type-check time.
    """

    def generate(
        self,
        prompts: Any,
        sampling_params: Any = None,
        *,
        use_tqdm: bool = False,
    ) -> list[Any]:
        ...


# ---------------------------------------------------------------------------
# Mock engine for smoke tests + CI
# ---------------------------------------------------------------------------

@dataclass
class _MockMetrics:
    """Mirror of vllm.sequence.RequestMetrics fields D consumes."""

    arrival_time: float
    first_scheduled_time: Optional[float]
    first_token_time: Optional[float]
    last_token_time: float
    time_in_queue: Optional[float] = None
    finished_time: Optional[float] = None


@dataclass
class _MockCompletionOutput:
    text: str
    token_ids: list[int] = field(default_factory=list)


@dataclass
class _MockRequestOutput:
    request_id: str
    prompt: str
    metrics: _MockMetrics
    outputs: list[_MockCompletionOutput]
    finished: bool = True


class MockEngine:
    """Deterministic fake that simulates vLLM's ``LLM`` for the metrics we read.

    Fakes:
      - arrival_time  → time.monotonic() at call entry
      - first_token_time → arrival + ttft_ms
      - last_token_time  → first_token + per_token_ms * num_output_tokens
      - generated text   → "mock output token N" repeated

    Does NOT actually sleep by default (would slow tests) — the timestamps are
    synthetic. Set ``simulate_realtime=True`` to actually block; useful when
    testing TimingContext behaviour.
    """

    def __init__(
        self,
        ttft_ms: float = 100.0,
        per_token_ms: float = 5.0,
        max_output_tokens: int = 32,
        simulate_realtime: bool = False,
    ) -> None:
        self.ttft_ms = ttft_ms
        self.per_token_ms = per_token_ms
        self.max_output_tokens = max_output_tokens
        self.simulate_realtime = simulate_realtime
        self.call_count = 0

    def generate(
        self,
        prompts: Any,
        sampling_params: Any = None,
        *,
        use_tqdm: bool = False,
    ) -> list[_MockRequestOutput]:
        if isinstance(prompts, str):
            prompts = [prompts]

        max_tokens = self._extract_max_tokens(sampling_params)
        n_tokens = min(self.max_output_tokens, max_tokens)

        results: list[_MockRequestOutput] = []
        for i, prompt in enumerate(prompts):
            arrival = time.monotonic()
            if self.simulate_realtime:
                time.sleep(self.ttft_ms / 1000.0)
            first_token = arrival + (self.ttft_ms / 1000.0)

            decode_seconds = n_tokens * self.per_token_ms / 1000.0
            if self.simulate_realtime:
                time.sleep(decode_seconds)
            last_token = first_token + decode_seconds

            text = " ".join(f"tok{j}" for j in range(n_tokens))
            results.append(
                _MockRequestOutput(
                    request_id=f"mock-{self.call_count}-{i}",
                    prompt=prompt,
                    metrics=_MockMetrics(
                        arrival_time=arrival,
                        first_scheduled_time=arrival,
                        first_token_time=first_token,
                        last_token_time=last_token,
                        time_in_queue=0.0,
                        finished_time=last_token,
                    ),
                    outputs=[
                        _MockCompletionOutput(
                            text=text,
                            token_ids=list(range(n_tokens)),
                        )
                    ],
                )
            )
            self.call_count += 1
        return results

    @staticmethod
    def _extract_max_tokens(sampling_params: Any) -> int:
        """Pull max_tokens off a SamplingParams or dict; default to 64."""
        if sampling_params is None:
            return 64
        if isinstance(sampling_params, dict):
            return int(sampling_params.get("max_tokens", 64))
        return int(getattr(sampling_params, "max_tokens", 64))


# ---------------------------------------------------------------------------
# Real engine factory — lazy vLLM import
# ---------------------------------------------------------------------------

def build_retention_config(retention_dict: dict[str, Any]) -> Any:
    """Construct a ``src.retention.config.RetentionConfig`` from a parsed YAML
    dict. Returns ``None`` if retention is disabled or absent.

    The dict shape matches ``configs/retention.yaml`` and Daksh's dataclasses:

        retention:
          enabled: true
          ttl: {alpha, default_ttl, safety_factor, use_per_tool_ema, use_ema}
          pin_manager: {max_pinned_fraction}
    """
    if not retention_dict or not retention_dict.get("enabled", False):
        return None
    from src.retention.config import (
        PinManagerConfig,
        RetentionConfig,
        TTLConfig,
    )
    ttl_kwargs = retention_dict.get("ttl", {}) or {}
    pin_kwargs = retention_dict.get("pin_manager", {}) or {}
    return RetentionConfig(
        enabled=True,
        ttl=TTLConfig(**{k: v for k, v in ttl_kwargs.items()
                         if k in {"alpha", "default_ttl", "safety_factor",
                                  "use_per_tool_ema", "use_ema"}}),
        pin_manager=PinManagerConfig(
            **{k: v for k, v in pin_kwargs.items()
               if k in {"max_pinned_fraction"}}
        ),
    )


def build_real_engine(cfg: dict[str, Any]) -> EngineProtocol:
    """Construct a real vLLM ``LLM`` from a parsed config dict.

    Imports vLLM lazily so unit tests + dry-run never pay the cost. Raises a
    clear error if vLLM is missing.

    Retention is wired through the ``retention_config`` kwarg added by the
    Workstream A patch to ``vllm/entrypoints/llm.py`` and
    ``vllm/engine/llm_engine.py::from_engine_args``. When
    ``cfg["engine"]["retention"]["enabled"]`` is true, this constructs a
    ``RetentionConfig`` from the YAML and forwards it; PinManager is then
    instantiated inside ``LLMEngine.__init__``.

    Quantization (Workstream B) and torch.compile (Workstream C) knobs are
    documented but presently no-op until those workstreams land their engine
    surface.
    """
    try:
        from vllm import LLM
    except ImportError as e:
        raise SystemExit(
            "vLLM is not importable. Install the vendored copy:\n"
            "  cd vllm && pip install -e .\n"
            f"(import error: {e})"
        ) from e

    model_cfg = cfg.get("model", {})
    engine_cfg = cfg.get("engine", {})

    llm_kwargs: dict[str, Any] = {
        "model": model_cfg.get("name"),
        "dtype": model_cfg.get("dtype", "auto"),
    }
    if "max_model_len" in model_cfg:
        llm_kwargs["max_model_len"] = model_cfg["max_model_len"]
    if "gpu_memory_utilization" in model_cfg:
        llm_kwargs["gpu_memory_utilization"] = model_cfg["gpu_memory_utilization"]

    # Quantization (Workstream B) — config slot defined; engine support TODO.
    quant = engine_cfg.get("quantization", {}).get("kv_cache")
    if quant is not None:
        # When B lands, this should set kv_cache_dtype or similar.
        llm_kwargs["kv_cache_dtype"] = quant

    # torch.compile (Workstream C) — engine flag, currently env-driven in vLLM.
    if engine_cfg.get("torch_compile", {}).get("enabled"):
        # vLLM v0.6.4 exposes torch.compile via VLLM_USE_TORCH_COMPILE env var.
        # Setting the env var is Workstream C's responsibility.
        pass

    # Retention (Workstream A) — forward RetentionConfig if enabled.
    retention_config = build_retention_config(engine_cfg.get("retention", {}))
    if retention_config is not None:
        llm_kwargs["retention_config"] = retention_config

    return LLM(**{k: v for k, v in llm_kwargs.items() if v is not None})


# ---------------------------------------------------------------------------
# Helpers used by run_eval.run_trace
# ---------------------------------------------------------------------------

def extract_request_metrics(output: Any) -> dict[str, Optional[float]]:
    """Pull TTFT, queue, and decode timing off a RequestOutput.

    Works for both vLLM's real RequestOutput and our _MockRequestOutput because
    both expose ``.metrics`` with the same field names.
    """
    m = output.metrics
    arrival = float(m.arrival_time)
    first_token = getattr(m, "first_token_time", None)
    last_token = getattr(m, "last_token_time", None)
    first_scheduled = getattr(m, "first_scheduled_time", None)

    ttft_ms = (first_token - arrival) * 1000.0 if first_token is not None else None
    queue_ms = (
        (first_scheduled - arrival) * 1000.0
        if first_scheduled is not None
        else None
    )
    decode_ms = (
        (last_token - first_token) * 1000.0
        if first_token is not None and last_token is not None
        else None
    )
    return {"ttft_ms": ttft_ms, "queue_ms": queue_ms, "decode_ms": decode_ms}


def output_token_count(output: Any) -> int:
    """Count tokens in the first completion of a RequestOutput. 0 if absent."""
    completions = getattr(output, "outputs", None)
    if not completions:
        return 0
    return len(getattr(completions[0], "token_ids", []) or [])
