#!/usr/bin/env python3
"""Daksh's KV-retention microbenchmark — A → B → C → A pattern.

Reference isolation test that demonstrates retention's value in a precisely
single-threaded sequence. This is the test that produced the 7.9× headline
on H100 (843 ms PC-only vs 107 ms retention).

Mechanism:
  1. Agent A submits a request → KV blocks land in the prefix cache
     (with retention: blocks are pinned for TTL_A * safety_factor seconds).
  2. Agent B submits N filler requests sequentially → cache fills; without
     retention A's blocks are LRU candidates.
  3. Agent C submits one request → forces eviction. Without retention,
     A is the unique LRU victim (oldest, idle). With retention, A's
     blocks are pinned, so a B filler is evicted instead.
  4. Agent A submits a SECOND request with the same prompt. Without
     retention: full prefill (cache miss). With retention: cache hit.

Reported metric: TTFT delta on A's second request.

Sizing invariants the script validates before running:
  C1: ba + bb        ≤ pool   (B alone does NOT evict A)
  C2: ba + bb + bc   > pool   (C forces eviction)
  C3: bb + bc        ≤ pool   (without A, B+C fit alone)

Usage:
    PYTHONPATH=/home/$USER/MoM python3 scripts/daksh_microbench.py
    PYTHONPATH=. python3 scripts/daksh_microbench.py --model meta-llama/Meta-Llama-3-8B
"""
import argparse
import gc
import math
import re
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                   help="HF repo id. Daksh's reference uses -Instruct; base "
                        "model also works for the cache-effect test.")
    p.add_argument("--max-len", type=int, default=4096,
                   help="max_model_len. Daksh: 4096 (gives ba+bb+bc just over pool).")
    p.add_argument("--ttl-a", type=float, default=1.0,
                   help="TTL seconds for retention. Effective TTL = ttl_a * 1.5 "
                        "via safety_factor. Must exceed B+C generation time.")
    p.add_argument("--frac-a", type=float, default=0.70,
                   help="Fraction of pool A occupies.")
    p.add_argument("--frac-b", type=float, default=0.25,
                   help="Per-filler fraction of pool. ba + bb ≤ 1.0.")
    p.add_argument("--frac-c", type=float, default=0.10,
                   help="Fraction of pool C occupies. ba + bb + bc > 1.0 forces eviction.")
    p.add_argument("--max-toks", type=int, default=16,
                   help="max_tokens for SamplingParams. Keep small to make the "
                        "phase B+C run quickly (so retention's TTL covers it).")
    p.add_argument("--out", default="results/isolation",
                   help="Where to write retention events.jsonl.")
    return p.parse_args()


def main():
    args = parse_args()
    BLOCK_SIZE = 16
    PAD = "The quick brown fox jumps over the lazy dog. " * 80

    # ── KV pool sizing ──────────────────────────────────────────────────
    import torch
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    m = re.search(r"(\d+\.?\d*)b", args.model.lower())
    model_gb = float(m.group(1)) * 2 if m else 14.0
    target_blocks = args.max_len // BLOCK_SIZE
    bpb = 2 * 32 * 8 * 128 * BLOCK_SIZE * 2  # Llama-3-8B: 32L, 8 KV-heads, 128-dim, fp16
    kv_gb = target_blocks * bpb / 1e9
    gpu_util = min(0.97, (model_gb + kv_gb + 2.0) / vram_gb)

    ta, tb, tc = [int(f * args.max_len) for f in (args.frac_a, args.frac_b, args.frac_c)]
    ba, bb, bc = [math.ceil(t / BLOCK_SIZE) for t in (ta, tb, tc)]

    print(f"GPU: {torch.cuda.get_device_name(0)}  ({vram_gb:.0f} GB)")
    print(f"Model: {args.model}")
    print(f"KV pool: {target_blocks} blocks  gpu_util={gpu_util:.3f}")
    print(f"Tokens A/B/C: {ta}/{tb}/{tc}   Blocks A/B/C: {ba}/{bb}/{bc}")

    # ── Validate isolation conditions ───────────────────────────────────
    print("\nValidation:")
    checks = [
        ("C1 A+B <= pool",    ba + bb <= target_blocks,
         f"{ba}+{bb}={ba + bb} <= {target_blocks}"),
        ("C2 A+B+C > pool",   ba + bb + bc > target_blocks,
         f"{ba + bb + bc} > {target_blocks}  (C forces eviction)"),
        ("C3 B+C <= pool",    bb + bc <= target_blocks,
         f"{bb + bc} <= {target_blocks}  (no pressure w/o pin)"),
    ]
    ok = True
    for label, passed, desc in checks:
        print(f"  {'✓' if passed else '✗'}  {label}: {desc}")
        if not passed:
            ok = False
    if not ok:
        print("\nABORT: fracs must sum > 1.0 (default 0.70+0.25+0.10=1.05 works).")
        sys.exit(1)
    print("  ✓ Conditions satisfied\n")

    # ── Prompts ─────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    def mkprompt(n, prefix=""):
        ids = tok.encode(prefix, add_special_tokens=False)
        chunk = tok.encode(PAD, add_special_tokens=False)
        while len(ids) < n:
            ids += chunk[:n - len(ids)]
        return tok.decode(ids[:n], skip_special_tokens=True)

    prompt_a = mkprompt(ta, "[SYS] You are helpful. [CTX] ")
    prompt_c = mkprompt(tc, "[Q] What is 2+2? ")

    def cache_state(llm):
        from vllm.core.block.interfaces import Device
        gpu = (llm.llm_engine.scheduler[0]
                  .block_manager.block_allocator._allocators[Device.GPU])
        return (gpu._hashless_allocator.get_num_free_blocks(),
                gpu.evictor.num_blocks)

    # ── Run one config ──────────────────────────────────────────────────
    def run(retention):
        from vllm import LLM, SamplingParams

        # Configure event logging if src.retention is importable.
        try:
            from src.retention import events as ev
            if ev._log_file:
                ev._log_file.close()
            ev._log_file = ev._wandb = None
            Path(args.out).mkdir(parents=True, exist_ok=True)
            ev.configure(
                log_file=f"{args.out}/events_{'ret' if retention else 'pc'}.jsonl",
                use_wandb=False,
            )
        except ImportError:
            pass

        kw = dict(model=args.model, dtype="float16", max_model_len=args.max_len,
                  gpu_memory_utilization=gpu_util, enable_prefix_caching=True)
        if retention:
            from src.retention.config import RetentionConfig, TTLConfig, PinManagerConfig
            kw["retention_config"] = RetentionConfig(
                enabled=True,
                ttl=TTLConfig(default_ttl=args.ttl_a, safety_factor=1.5, alpha=0.3),
                pin_manager=PinManagerConfig(max_pinned_fraction=0.85),
            )
        try:
            llm = LLM(**kw)
        except TypeError as e:
            print(f"  [TypeError — falling back without retention_config]: {e}")
            kw.pop("retention_config", None)
            llm = LLM(**kw)

        # Diagnostics: prove which vLLM is loaded and whether pin_manager exists.
        print(f"  retention_config in kw: {'retention_config' in kw}")
        print(f"  LLMEngine file: {__import__('inspect').getfile(llm.llm_engine.__class__)}")
        try:
            import src.retention.events  # noqa: F401
            print("  src.retention: OK")
        except ImportError as e:
            print(f"  src.retention: MISSING — {e}")
            print("  Run with: PYTHONPATH=/home/$USER/MoM python3 scripts/daksh_microbench.py")

        pm = getattr(llm.llm_engine, 'pin_manager', None)
        pm_in_sched = getattr(llm.llm_engine.scheduler[0], 'pin_manager', None)
        print(f"  pin_manager (engine):    {pm}")
        print(f"  pin_manager (scheduler): {pm_in_sched}")

        actual_blocks = llm.llm_engine.cache_config.num_gpu_blocks
        print(f"  ACTUAL KV blocks: {actual_blocks}  (target was {target_blocks})")

        sp = SamplingParams(max_tokens=args.max_toks, temperature=0.0)

        # ── Phase 1: Agent A (first request) ────────────────────────────
        t0 = time.monotonic()
        llm.generate(
            prompt_a, sp, use_tqdm=False, program_id="agent_a",
            is_tool_call_pending=retention,
            tool_name="tool_x" if retention else None,
        )
        a_done = time.monotonic()
        print(f"  A_req1: {(a_done - t0) * 1e3:.0f} ms  pinned={retention}")
        tf, ev_n = cache_state(llm)
        print(f"    → truly_free={tf}  in_evictor={ev_n}")

        # ── Phase 2: Agent B (N fillers, sequential) ────────────────────
        t_b = time.monotonic()
        actual_blocks = llm.llm_engine.cache_config.num_gpu_blocks
        bb_actual = math.ceil(tb / BLOCK_SIZE)
        n_fillers = math.ceil((actual_blocks * 1.1) / bb_actual)
        print(f"  actual KV pool: {actual_blocks} blocks, "
              f"running {n_fillers} fillers ({bb_actual} blocks each)")

        for i in range(n_fillers):
            p = mkprompt(tb, prefix=f"[FILLER_{i}] context_{i}. ")
            llm.generate(p, sp, use_tqdm=False)
            tf, ev_n = cache_state(llm)
            print(f"  filler {i + 1}/{n_fillers}  truly_free={tf}  in_evictor={ev_n}")

        print(f"  B total: {(time.monotonic() - t_b) * 1e3:.0f} ms")

        # ── Phase 3: Agent C ────────────────────────────────────────────
        t_c = time.monotonic()
        llm.generate(prompt_c, sp, use_tqdm=False)
        elapsed = time.monotonic() - a_done
        print(f"  C: {(time.monotonic() - t_c) * 1e3:.0f} ms   "
              f"(A→C elapsed: {elapsed:.1f}s, TTL={args.ttl_a}s "
              f"× safety_factor=1.5 = {args.ttl_a * 1.5:.1f}s)")
        tf, ev_n = cache_state(llm)
        print(f"    → truly_free={tf}  in_evictor={ev_n}")
        if retention and elapsed > args.ttl_a * 1.5:
            print("  [WARN] A's pin may have expired — increase --ttl-a")

        # ── Phase 4: Agent A again (KEY measurement) ────────────────────
        tf, ev_n = cache_state(llm)
        print(f"    → before A_req2: truly_free={tf}  in_evictor={ev_n}")
        t0 = time.monotonic()
        out = llm.generate(prompt_a, sp, use_tqdm=False, program_id="agent_a")
        tf, ev_n = cache_state(llm)
        print(f"    → after  A_req2: truly_free={tf}  in_evictor={ev_n}")
        m = out[0].metrics
        print(f"  first_token_time set: {m.first_token_time is not None}")

        if getattr(m, "first_token_time", None):
            ttft = (m.first_token_time - m.arrival_time) * 1e3
        else:
            ttft = (time.monotonic() - t0) * 1e3
        print(f"  A_req2: {ttft:.1f} ms TTFT  ← KEY")

        del llm
        gc.collect()
        torch.cuda.empty_cache()
        return ttft

    # ── Run both configs ────────────────────────────────────────────────
    print("=== prefix_cache_only ===")
    pc = run(retention=False)

    import json
    ef = Path(f"{args.out}/events_pc.jsonl")
    if ef.exists():
        evs = [json.loads(l) for l in ef.read_text().splitlines() if l]
        for t in ["pin", "reuse", "evict", "expire", "pin_rejected_budget"]:
            print(f"  {t}: {sum(1 for e in evs if e['event_type'] == t)}")
    else:
        print(f"  no events file at {ef} — src.retention not on PYTHONPATH "
              "or events not configured")

    print("\n=== retention (pin/evict) ===")
    ret = run(retention=True)

    ef = Path(f"{args.out}/events_ret.jsonl")
    if ef.exists():
        evs = [json.loads(l) for l in ef.read_text().splitlines() if l]
        for t in ["pin", "reuse", "evict", "expire", "pin_rejected_budget"]:
            count = sum(1 for e in evs if e['event_type'] == t)
            detail = [(e.get('num_blocks'), e.get('ttl_assigned'))
                      for e in evs if e['event_type'] == t]
            print(f"  {t}: {count}  {detail[:5]}{'...' if len(detail) > 5 else ''}")
    else:
        print(f"  no events file at {ef}")

    # ── Result ──────────────────────────────────────────────────────────
    speedup = pc / ret if ret > 0 else float("inf")
    print(f"""
{'=' * 50}
RESULT: A_req2 TTFT
  prefix_cache_only : {pc:7.1f} ms  (A evicted by C → recompute)
  retention         : {ret:7.1f} ms  (A pinned → cache hit)
  Speedup           : {speedup:.1f}x
{'=' * 50}""")
    if speedup < 1.2:
        print("  No clear benefit. Check:")
        print("  • PYTHONPATH includes ~/MoM (so src.retention is importable)")
        print("  • LLMEngine file is from ~/MoM/vllm/, NOT site-packages")
        print("  • events_ret.jsonl shows pin events > 0")
        print("  • Increase --ttl-a if A→C elapsed exceeded TTL")


if __name__ == "__main__":
    main()
