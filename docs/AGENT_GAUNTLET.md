# Agent Tool Gauntlet — 4 Colonel models × EDR fan-out

## Purpose
The OS AGI self-improvement layer needs *evidence*: every model, run through the
same agent-tool battery, with every action recorded into the WuBuOS EDR layer so
the OS can audit, replay, and score the agents. This is the generator that
produces that evidence.

## Components (all C11, no third-party deps)
- `tools/agent_gauntlet/agent_gauntlet.h` — public API
- `tools/agent_gauntlet/agent_gauntlet.c` — model registry (4 Colonels), fixture
  fallback, gauntlet loop, EDR fan-out, scoring
- `tools/agent_gauntlet/gauntlet_run.c` — CLI driver + leaderboard
- `tools/agent_gauntlet/test_gauntlet.c` — regression gate

## The four Colonel models
On-disk checkpoints live **persistently** under `/home/wubu/models/<Name>/`
(multi-shard safetensors + tokenizer.json + config.json). `wubu_model_init_auto`
accepts the **directory** (dense/moe) or the **adapter file** (BTL-3, with
`BTL_BASE` = the Qwen3.6-27B base dir). The canonical, exact-dim inventory is
stored on HuggingFace:
**https://huggingface.co/datasets/WaefreBeorn/bytropix-colonel-registry**
(`registry.json` + card).

| codename        | kind | on-disk checkpoint (resolved at runtime) |
|-----------------|------|------------------------------------------|
| Qwen3.6-27B     | dense (BTL-3 base) | `/home/wubu/models/Qwen3.6-27B/` (15 shards) |
| Agents-A1-4B    | dense hybrid | `/home/wubu/models/Agents-A1-4B/` (2 shards) |
| KAT-Coder-V2.5  | MoE 256/8    | `/home/wubu/models/KAT-Coder-V2.5-Dev/` (13 shards) |
| BTL-3           | LoRA on Qwen3.6-27B | `/home/wubu/models/BTL-3/adapter_model.safetensors` (+ `BTL_BASE`) |

Exact dims (hidden/layers/heads/experts) are in the HF registry, not hardcoded
here. Agents-A1-4B carries the verified repetition/DRY tuning
(rp 1.05/1.1, dry 0.5/1.2, dry-base 1.75, ctx 131072, ub 2048) wired into
`gen_text` defaults.

Missing/oversized checkpoints **fall back to `fixture_model.safetensors`** so the
harness always runs + verifies on this box (13 GB RAM). When the real 9 GB+
weights are present, `wubu_model_init_auto` loads them by their real tensor
shapes (dimension-driven forward).

## The three tools (agent-task battery)
1. **shell** — emit a `cmd:` line that prints the hostname
2. **file**  — emit a `cmd:` heredoc that writes `hello wubu` to `/tmp/gauntlet_out.txt`
3. **code**  — emit a `note:` one-line summary of what `wubu_ssm_forward` does

## EDR fan-out (WuBuOS AGI self-improvement layer)
Every decode step fans the token sample as an `EDR_EV_AGENT_ACTION` (type=26)
via `edr_log_agent_action(EDR_AGENT_TYPE, ...)`. When a model emits a recognized
tool-form, a second dedicated agent-action event records `model=… task=… tool-form=…`.
The EDR ring is the SAME lock-free queue the behavioral modules drain (fanotify,
proc_pin, poller) — so agent actions are observable on exactly the same audit
surface as OS process/file/network telemetry. The OS self-improvement loop
consumes `edr_recent_events()` for replay + scoring.

Build: `make gauntlet` (driver) / `make test_gauntlet` (regression).
Links bytropix engine (`CORE_OBJ`) + WuBuOS `$(EDR_SRC)` (no daemon) + `-lpthread`.
Override the WuBuOS checkout path with `make WUBUOS=/path/to/wubuos test_gauntlet`.

## Verification (this box, fixture fallback)
```
make test_gauntlet
  ok: EDR engine starts
  ok: all four model slots loaded (fixture fallback)
  ok: gauntlet fanned EDR agent actions (total>0)
  ok: every (model,task) decoded (n_actions>0)
  ok: EDR recent-events snapshot returns fanned actions
  ALL GAUNTLET CHECKS PASSED (models=4, total_actions=300, edr_events=9)
```
`correct=0` on the fixture is expected (no `tokenizer.json` → synthetic-id
fallback disables the decoded-text tool heuristic). On real weights with a
tokenizer, `correct` reflects whether the emitted tool-form achieves the goal.
