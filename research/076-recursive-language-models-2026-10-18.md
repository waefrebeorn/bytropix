# Recursive Language Models — context as an object, not a window

**Seed:** arXiv 2512.24601v3, *Recursive Language Models* (Alex L. Zhang, Tim
Kraska, Omar Khattab — MIT CSAIL), v1 31 Dec 2025, v3 11 May 2026.
**Licence: CC BY 4.0** — usable, unlike the previous seed's NC licence.

**Wave:** 7 layers × 25 + 25 convergence = **200 queries, 1,000 records, 933
unique URLs** at `/home/wubu/opt/research/rlm-2026-10-18/`. Only the seed was
read in full; layers 1–7 are discovery.

## The claim

A long prompt should not be copied into a fixed window. It should live as a
**variable in an external environment** (a REPL), and the model should write
code that peeks, slices, and recursively calls itself over it. The root model
sees only constant-size metadata (length, prefix, how to access); intermediate
results live in variables, not in the conversation.

Three design choices separate a real RLM from the scaffolds that look similar:

1. **A symbolic handle** to the prompt — never the prompt itself in the root context.
2. **No `Finish(value)` output path** — the final answer comes from a variable, so
   output length is not capped by the window.
3. **Symbolic recursion** — code running inside the environment can call the model
   in loops over slices, so the work can be Ω(|P|) or Ω(|P|²), not a handful of
   verbalized sub-tasks.

## Numbers (Table 1, GPT-5, recursion depth 3 unless noted)

| Task | Base GPT-5 | RLM | Claude Code (+offload) |
|---|---|---|---|
| CodeQA (23K–4.2M tok) | 24 (context-limited) | **58** | 62 |
| BrowseComp+ 1K docs (6–11M tok) | 0 (context-limited) | **92** | 84 |
| OOLONG (131K) | 44 | **58** | 48 |
| OOLONG-Pairs (32K) | 0.1 | **76** | 6.5 |

Depth matters most on the quadratic task: OOLONG-Pairs goes 43.9 (depth 0) → 58
(depth 1) → 65.5 (depth 2) → 76 (depth 3). Cost is comparable at the median
(GPT-5 depth 1: $0.11 CodeQA, $0.99 BrowseComp+) but the 95th percentile is a
long tail from sequential sub-calls.

**Post-training:** 750 LongBenchPro tasks → 2,250 candidate trajectories,
filtered to **1,000**; prime-rl, batch 64, 300 steps, **48 H100-hours**.
RLM-Qwen3-8B beats Qwen3-8B by a **median 28.3%** and runs **>3× faster**;
RLVR on MRCRv2 generalizes to the longer split.

## The useful failure modes

- Qwen3-8B uses **more sub-calls on incorrect trajectories** — recursion volume
  is not a quality signal, and a budget is not a strategy.
- Cost and runtime are long-tailed; early stop / timeout is mandatory.
- Base models are not merely worse, they are **context-limited**: several cells
  in Table 1 are `0.0` because the input never fit.

## What WuBu built

`include/wubu_rlm.h` + `src/wubu_rlm.c` — the controller, without pretending to
be a model. The leaf callback is where a model or a kernel routine goes.

- `wubu_rlm_run(data, len, cfg, leaf, merge, ctx, out, outcap, stats)`
- `cfg = {chunk, max_depth, budget}`; stats record leaves, nodes, merges, bytes
  covered, depth reached, truncation.
- Guarantees: each byte reaches at most one leaf; budget honoured exactly; merge
  order preserved; a leaf error aborts instead of returning a partial answer;
  output overflow returns `-2` rather than silently truncating.

`tools/test_rlm` — **18 checks, 0 failures**: 1 MiB at depth 8 → 256 leaves /
255 merges / every byte once; budget 100,000 → exactly 100,000 covered and
`truncated=1`; poisoned leaf → `-1`; order preserved (`0,1,2…`); tiny output →
`-2`.

## Why this is the OS shape, not just a prompt trick

The paper's environment is a Python REPL. WuBuOS already has a **namespace**
(9P, KV-FS), which is a stronger environment than a REPL: it is persistent,
permissioned, and visible to other processes. The natural WuBu form is:

```
root model sees:  /n/task/123  (length, mtime, a 256-byte head)
model writes:     read /n/task/123?offset=…&count=…, grep over it, recurse
sub-calls:        spawn on slices, results land as files, not chat text
```

That also closes the loop with the self-host ladder: if the machine owns its
storage, the context object is a real file with a real inode, and "context
management" becomes a filesystem operation instead of a summarization heuristic.

## Seven-hop convergence

1. **RLM** — prompt as environment variable; code, not prose, drives decomposition.
2. **CodeAct** — code as action, but still puts the prompt in the window; loses at scale.
3. **Compaction** — cheap and strong on QA, collapses on tasks needing every line.
4. **Retrieval agents (BM25)** — good when a few documents matter; useless on OOLONG.
5. **Context rot / RULER** — quality falls with length before the window is full.
6. **Inference-time scaling** — recursion is a compute-scaling axis, like reasoning.
7. **OS architecture** — a namespace makes the environment durable and shareable.

Convergence: **the winning designs move data out of the model and address it
symbolically.** Every hop that keeps the payload in the window hits the same wall.

## What to do next

1. **A read-only 9P gateway for the context object** — the model gets `open/read/seek`
   over `/n/ctx` instead of a pasted prompt. `wubu_rlm` already enforces the budget.
2. **A recursion budget in the decision ledger** — reuse `wubu_decision_ledger` to
   price sub-calls; the paper's long tail is exactly an unpriced recursion.
3. **Depth policy from the answer key** — OOLONG-Pairs needs depth 3; S-NIAH needs
   depth 0. Do not pay for depth the task cannot use.
4. **Do not chase the 28.3%** — that is a fine-tune of an 8B on 1,000 trajectories
   (48 H100-hours). The structural win (context as object) is available today;
   the model win is a later, optional step.

## Honest caveats

- One seed read in full; layers 1–7 are search discovery, not verified citations.
- The paper's strongest cells use GPT-5 (272K window); our hosted tests cannot
  reproduce them, which is why the module is a controller test, not a benchmark.
- Costs are the authors' provider pricing and vary by more than 2× across tasks.
