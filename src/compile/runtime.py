"""Runtime wiring for Workstream C compile/profiling knobs."""
from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

from src.compile.config import CompileConfig, ProfileConfig, WorkstreamCConfig


def configure_compile_environment(config: CompileConfig) -> dict[str, str]:
    """Set vLLM/PyTorch environment variables for torch.compile.

    vLLM 0.6.4 exposes compilation through ``VLLM_TORCH_COMPILE_LEVEL``. The
    phase target list is kept in metadata because this version does not provide
    separate public callables for prefill and decode compilation.
    """
    if not config.enabled:
        return {}

    env_updates = {
        "VLLM_TORCH_COMPILE_LEVEL": "1",
        "VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE": "1" if config.fullgraph else "0",
    }
    os.environ.update(env_updates)
    return env_updates


def configure_profile_environment(config: ProfileConfig) -> dict[str, str]:
    """Set vLLM profiler environment variables before the real engine starts."""
    if not config.enabled or not config.pytorch_profiler:
        return {}

    torch_dir = (config.output_dir / "torch").resolve()
    torch_dir.mkdir(parents=True, exist_ok=True)
    env_updates = {"VLLM_TORCH_PROFILER_DIR": str(torch_dir)}
    os.environ.update(env_updates)
    return env_updates


def configure_torch_compile_backend(config: CompileConfig) -> bool:
    """Forward the requested backend to vLLM's plugin registry if available."""
    if not config.enabled:
        return False
    try:
        from vllm.plugins import set_torch_compile_backend  # type: ignore
    except Exception:
        return False
    set_torch_compile_backend(config.backend)
    return True


def compile_metadata(config: WorkstreamCConfig) -> dict[str, Any]:
    compile_cfg = config.compile
    profile_cfg = config.profile
    metadata: dict[str, Any] = {
        "compile_enabled": compile_cfg.enabled,
        "compile_targets": list(compile_cfg.targets),
        "compile_backend": compile_cfg.backend,
        "compile_mode": compile_cfg.mode,
        "compile_dynamic": compile_cfg.dynamic,
        "compile_fullgraph": compile_cfg.fullgraph,
        "compile_warmup_iters": compile_cfg.warmup_iters,
        "profile_enabled": profile_cfg.enabled,
        "profile_pytorch_profiler": profile_cfg.pytorch_profiler,
        "profile_nsight": profile_cfg.nsight,
    }
    if compile_cfg.enabled:
        metadata["compile_hook"] = "vllm_model_runner"
        metadata["compile_phase_limitation"] = (
            "vLLM 0.6.4 exposes torch.compile at model execution level; "
            "prefill/decode are phase-tagged and measured separately, but not "
            "compiled as independent callables."
        )
    return metadata


@contextmanager
def maybe_profile(
    engine: Any,
    config: ProfileConfig,
) -> Iterator[Any | None]:
    """Start local and vLLM profiler sessions when explicitly enabled."""
    if not config.enabled or not config.pytorch_profiler:
        yield None
        return

    config.output_dir.mkdir(parents=True, exist_ok=True)
    profiler_cm = _local_torch_profiler(config)
    profiler = profiler_cm.__enter__()
    _try_engine_profile(engine, "start_profile")
    try:
        yield profiler
    finally:
        _try_engine_profile(engine, "stop_profile")
        profiler_cm.__exit__(None, None, None)


def record_phase(name: str) -> Any:
    """Return a profiler range context; no-op if torch is unavailable."""
    try:
        import torch  # type: ignore
    except Exception:
        return nullcontext()
    return torch.profiler.record_function(name)


def profiler_step(profiler: Any | None) -> None:
    if profiler is not None and hasattr(profiler, "step"):
        profiler.step()


def write_nsight_command(
    *,
    config: ProfileConfig,
    output_dir: Path,
    argv: list[str],
) -> Path | None:
    """Write the exact Nsight command to run on the GPU VM.

    The runner does not relaunch itself under ``nsys``; doing so would hide the
    user's actual command and make local/mock runs surprising.
    """
    if not config.enabled or not config.nsight:
        return None
    config.output_dir.mkdir(parents=True, exist_ok=True)
    script = output_dir / "nsight_command.sh"
    nsys_output = (config.output_dir / "nsight" / "mom_eval").resolve()
    nsys_output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--force-overwrite=true",
        "-o",
        str(nsys_output),
        *argv,
    ]
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" +
                      " ".join(_shell_quote(part) for part in command) + "\n")
    script.chmod(0o755)
    return script


def _local_torch_profiler(config: ProfileConfig) -> Any:
    try:
        import torch  # type: ignore
    except Exception:
        return nullcontext(None)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    local_dir = (config.output_dir / "local").resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    return torch.profiler.profile(
        activities=activities,
        record_shapes=config.record_shapes,
        profile_memory=config.profile_memory,
        with_stack=config.with_stack,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(local_dir)),
    )


def _try_engine_profile(engine: Any, method_name: str) -> None:
    method = getattr(engine, method_name, None)
    if method is None:
        return
    try:
        method()
    except RuntimeError as exc:
        if "Profiler is not enabled" not in str(exc):
            raise


def _shell_quote(value: str) -> str:
    if value and all(ch.isalnum() or ch in "/._-:=," for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"
