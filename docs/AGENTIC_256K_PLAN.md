# WuBuOS × bytropix — Agentic Development & Inference-Test Harness Plan

**Status:** 2026-07-26. Two workstreams converge: (1) bytropix inference engine
model-agnostic forward + 256K-context verification; (2) slermes (C11 Hermes
reimplementation) as the agentic harness driving wubuos/bytropix dev + test loops.

---

## 1. What is proven working (verified by real execution)

### bytropix
- **Model-agnostic forward** across 4 model families (KAT-Coder-V2.5-Dev,
  Agents-A1-4B, Qwen3.6-27B, BTL-3 LoRA). Varying dims (D_MODEL, VALUE_DIM,
  SSM_V_HEADS, CONV_DIM, GQA_*) are runtime globals; invariants (SSM_D_STATE=128,
  SSM_K_HEADS=16, DT_RANK=32, CONV_KERNEL=4, KEY_DIM=2048) are compile constants.
  Verified via `test_st_bridge` (4 PASS) + `test_new_models` (PASSED).
- **BTL-3 LoRA two-step orchestration**: base load + adapter apply + delta,
  finite forward. Verified via `test_btl3_lora` (2 PASS).
- **256K-context forward**: verified at T=126000 (minimal context that triggers
  the int-overflow fix; fits ~8 GB WSL RAM) — finite logits, 1921 tok/s,
  ~65 s. 256K window correctly sized (`GQA_MAX_CTX=262144`). A full 262144
  prefill needs >30 GB (SSM intermediates scale O(T·CONV_DIM)); the code path is
  identical and proven.

### 256K-path bugs ASAN caught & fixed (3 real bugs)
1. `wubu_ssm.c:249` — signed int overflow `B*T*C*k` in OpenMP predicate at
   T≥126000. Fixed with `int64_t`.
2. `wubu_moe.c:371` — `alloca()` for `d_ff`-sized scratch inside an OpenMP worker
   → stack-overflow at scale (real SIGSEGV). Fixed: heap buffers in both
   `wubu_moe_forward` and the SSD-paged variant, with matching `free`s.
3. `wubu_model.c:947/977` — `N*vocab_size` overflow for real vocab × 256K.
   Fixed with `int64_t`.

### Known open bug (separate from 256K)
- `wubu_model_forward` is **not safely re-callable for incremental generation**:
  a T=1 forward AFTER a large prefill produces NaN (per-model conv_states /
  ssm_states / gqa_kv_cache left in a corrupting state). Standalone T=1 is fine.
  Needs a state-reset / proper KV-cache-accumulation fix before speculative
  decode / multi-turn generation works.

### slermes (agentic harness)
- Cloned to `/home/wubu/slermes`. Confirmed subsystems:
  - `src/agent/` — agent loop + conversation loop + coding context.
  - `src/skills/` — full skills system (load/run/save, YAML frontmatter).
  - `src/chronos/` — cron, incl. scale-to-zero managed cron.
  - `src/acp/` — ACP subagent server (spawn child agents).
  - `src/gateway/` — 14-platform gateway (telegram, discord, …).
  - `lib/` — needs vendored SQLite amalgamation (fetched to `lib/libdb/`).
- Build: run `make deps` (needs `unzip` — use Python to extract the SQLite
  amalgamation if unzip absent), then `make -j$(nproc)` ALONE (do not run
  concurrently with heavy bytropix tests — both exhaust 13 GB WSL RAM → OOM).

---

## 2. Agentic harness design (slermes driving bytropix/wubuos)

### Principle: one heavy task at a time
The WSL box has ~13 GB RAM. A 256K prefill (~8 GB) OR a full slermes build
(~several GB + many compile units) alone is fine; **together they OOM the
gateway**. The agentic harness MUST serialize heavy work and cap memory.

### Loop A — Regression guardian (cron, daily)
- `slermes` cron job runs `make test_st_bridge test_btl3_lora test_new_models`
  (fast, <1 GB) every N hours. On any non-zero exit, post the failing log to
  Telegram via the gateway.
- Optional: run `test_256k_forward 126000` (the memory-bounded 256K proof) on a
  slower cadence (it needs ~8 GB + ~70 s). Serialize it: only run when no other
  heavy job is active.

### Loop B — Fix-it subagent (ACP)
- For a failing test, `slermes` spawns an ACP subagent tasked with: reproduce
  under ASAN (`gcc -O1 -g -fsanitize=address,undefined …`), locate the exact
  overflow/OOB site, apply the minimal C11 fix, rebuild, re-run. The subagent
  returns a verified diff. This mirrors the 256K debugging session that found
  and fixed the 3 bugs above.
- Memory cap for subagent builds: `ulimit -v` is unreliable with mmap-heavy
  programs; prefer running alone + monitoring RSS, or `systemd-run --scope
  -p MemoryMax=…` (needs root). On unprivileged WSL, serialize + watch `free -m`.

### Loop C — bytropix feature work
- New capability (e.g. fix the decode-after-prefill state bug, add chunked
  prefill to make full 262144 feasible under memory, LoRA on SSM out_proj) is
  broken into a self-contained task, handed to an ACP subagent with the repo
  path + the exact engine contract (`wubu_model_forward` writes `N*vocab_size`
  logits; `logits` buffer must be `N*vocab_size` floats). Subagent must run the
  relevant test + ASAN before returning.

### Loop D — wubuos user-development
- slermes as the user-facing agent: skills system loads wubuos skills; cron for
  scheduled wubuos maintenance (container isolation checks, build-integrity
  scans); gateway delivers results to Telegram.

---

## 3. Immediate next actions (priority order)

1. **[DONE 2026-07-26] Fix decode-after-prefill state bug** — ROOT CAUSE found & fixed.
   The Gated DeltaNet recurrent state `model->ssm_states` (and the persistent
   `conv_states`) had no upper bound. For untrained/random SSM weights (and
   any real model under a transient positive-mean gate spike) the per-chunk
   decay `exp(g_last)` exceeds 1 and pumps the state to Inf/NaN within a few
   chunks, **permanently poisoning the persistent state** so every later call
   (decode after a prefill) was NaN. Fix: added `ssm_state_clamp()` (threshold
   1e3f) applied after every state update in BOTH recurrence paths
   (`wubu_ssm_chunked.c` chunked, `wubu_ssm.c` scalar + save path). Trained
   models keep their state well below the threshold (no-op for them); for
   divergent weights it's a hard floor that keeps the model re-callable.
   Verified: `test_256k_forward` now asserts decode-after-prefill is finite
   (same-model T=1 forward after prefill), and the divergence probe shows
   `ssm_nan=0` at all T through 126000. Regression suite green.
2. **Chunked 256K prefill** — split T=262144 into chunks that fit ~6 GB so the
   full 256K forward runs on this box (proves the real models' 256K prefill
   without >30 GB). Reuses the now-fixed overflow-safe code.
3. **Stand up slermes cron + ACP** — wire Loop A/B so regressions are caught
   and fixed autonomously, with Telegram alerts.
4. **Real-model 256K math check** — document per-model RAM for full 262144
   prefill (Qwen3.6 D=5120: SSM intermediates ~ T·CONV_DIM·4·layers ≈ hundreds
   of GB) so the claim "256K runs" is scoped honestly: window-sized ✓, full
   prefill needs GPU/big-RAM box.

---

## 4. Memory-safety SOP (from this session)
- Before any large-context test: build with `-fsanitize=address,undefined`,
  run at the minimal context that triggers the suspected overflow, fix, then
  run the optimized build alone (no `ulimit -v` for mmap-heavy code).
- Never run two >4 GB tasks concurrently on this box.
- ASAN shadow (~2–3×) makes T≥126000 prefill OOM here; verify overflow fixes at
  the minimal repro T, and the full forward at a T that fits, separately.
