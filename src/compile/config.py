"""Config parsing for Workstream C.

The project historically used ``engine.torch_compile.enabled`` as a placeholder.
This module keeps that shape working while also accepting the proposal's
top-level ``compile`` and ``profile`` blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


_VALID_TARGETS = {"prefill", "decode"}


@dataclass(frozen=True)
class CompileConfig:
    enabled: bool = False
    targets: tuple[str, ...] = ("prefill", "decode")
    backend: str = "inductor"
    mode: str = "default"
    dynamic: bool = False
    fullgraph: bool = False
    warmup_iters: int = 1


@dataclass(frozen=True)
class ProfileConfig:
    enabled: bool = False
    pytorch_profiler: bool = False
    nsight: bool = False
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    output_dir: Path = Path("results/profiles")


@dataclass(frozen=True)
class WorkstreamCConfig:
    compile: CompileConfig = field(default_factory=CompileConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)


def load_workstream_c_config(cfg: Mapping[str, Any] | None) -> WorkstreamCConfig:
    """Parse compile/profile settings from a full evaluation config dict."""
    cfg = cfg or {}
    engine = _mapping(cfg.get("engine"))

    compile_blob = _mapping(cfg.get("compile"))
    if not compile_blob:
        compile_blob = _mapping(engine.get("compile"))
    if not compile_blob:
        compile_blob = _mapping(engine.get("torch_compile"))

    profile_blob = _mapping(cfg.get("profile"))
    if not profile_blob:
        profile_blob = _mapping(engine.get("profile"))

    return WorkstreamCConfig(
        compile=_parse_compile_config(compile_blob),
        profile=_parse_profile_config(profile_blob),
    )


def _parse_compile_config(blob: Mapping[str, Any]) -> CompileConfig:
    targets = _normalize_targets(blob.get("targets", ("prefill", "decode")))
    warmup_iters = int(blob.get("warmup_iters", 1))
    if warmup_iters < 0:
        raise ValueError("compile.warmup_iters must be >= 0")
    return CompileConfig(
        enabled=bool(blob.get("enabled", False)),
        targets=targets,
        backend=str(blob.get("backend", "inductor")),
        mode=str(blob.get("mode", "default")),
        dynamic=bool(blob.get("dynamic", False)),
        fullgraph=bool(blob.get("fullgraph", False)),
        warmup_iters=warmup_iters,
    )


def _parse_profile_config(blob: Mapping[str, Any]) -> ProfileConfig:
    enabled = bool(blob.get("enabled", False))
    pytorch_profiler = bool(blob.get("pytorch_profiler", False))
    nsight = bool(blob.get("nsight", False))
    return ProfileConfig(
        enabled=enabled,
        pytorch_profiler=pytorch_profiler,
        nsight=nsight,
        record_shapes=bool(blob.get("record_shapes", True)),
        profile_memory=bool(blob.get("profile_memory", True)),
        with_stack=bool(blob.get("with_stack", False)),
        output_dir=Path(str(blob.get("output_dir", "results/profiles"))),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalize_targets(raw_targets: Any) -> tuple[str, ...]:
    if isinstance(raw_targets, str):
        targets = (raw_targets,)
    else:
        targets = tuple(str(t) for t in raw_targets)
    unknown = set(targets) - _VALID_TARGETS
    if unknown:
        raise ValueError(
            "compile.targets must contain only 'prefill' and/or 'decode'; "
            f"got {sorted(unknown)}"
        )
    return targets
