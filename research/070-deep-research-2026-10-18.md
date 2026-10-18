# 070 — Deep research wave 2026-10-18: what the labs shipped, and what it changes for us

Method: Kevin-Bacon chain at 25 per layer × 7 layers, aggregated, then a second
25-deep pass on the convergence (200 queries total). Every arXiv ID was
re-verified against the arXiv API: **432/432 real, 430 PDFs obtained**.
Raw layers: `research/deep-2026-10-18/`. PDFs:
`/home/wubu/research-archive/deep-2026-10-18/papers/`.

This doc records only what the verified abstracts actually say. Anything not
verified is marked as a hypothesis.

---

## 1. The lab landscape (verified technical reports)

| lab | paper | what is actually new |
|---|---|---|
| Google DeepMind | **Gemma 4** `2607.02770` (2026-07-02) | dense **and** MoE, 2.3B–31B; a **unified encoder-free 12B** that ingests raw audio and image patches; thinking mode; inference-speed work |
| Alibaba Qwen | **Qwen3-VL** `2511.21631` | 256K interleaved context; dense 2B/4B/8B/32B + MoE 30B-A3B / 235B-A22B |
| Alibaba Qwen | **Qwen3-Coder-Next** `2603.00729` | 80B total / **3B active**; agentic training on synthesized *verifiable* coding tasks with executable environments |
| DeepSeek | **DeepSeek-V3** `2412.19437` | 671B/37B active; MLA; DeepSeekMoE; **auxiliary-loss-free** load balancing; multi-token prediction |
| OpenAI | **GPT-5 System Card** `2601.03267` | one system with a **real-time router** (fast vs deep model) trained on switch/preference/correctness signals |
| Mistral | **Ministral 3** `2601.08584` | 3B/8B/14B × {base, instruct, reasoning}; **Cascade Distillation** (iterative prune + distill) |
| Tencent | **Hunyuan-A13B** `2609.27284` | 80B total / **13B active**; 20T-token corpus; dual-mode CoT |
| Apple | **vllm-mlx** `2601.19139` | native MLX runtime; **21–87 %** higher text throughput than llama-cpp; continuous batching |
| Apple | **ANE MoE** `2604.18788` | why MoE is hard on NPUs: dynamic routing shapes, top-k/scatter, many tiny kernel launches |
| NVIDIA / misc | 119 inference papers | KV compression, speculative drafting, serving schedulers |

Reading of the trend: **sparse activation is now the default** (3B/80B, 13B/80B,
30B-A3B), **routing is becoming a first-class runtime component** (GPT-5's
router, DeepSeek's loss-free balancing), and **the encoder is being dissolved**
(Gemma 4's encoder-free 12B). All three land on decisions we already made:
our MoE router slot, the single-encoder modality-agnostic base, and the
physics router as a swappable selector.

---

## 2. What is directly actionable for our open goals

### COMPILER #37 — recursive-call live-range spill (fib(10) = 55)

The verified literature on the mechanism is `2011.05608` *Efficient global
register allocation* (2020-11-11): graph coloring and linear scan both pay
heavily for liveness computation, and **neither models spill cost across call
boundaries well**.

Our bug was exactly a boundary bug: the JIT's flat memory had no per-call frame,
so a value live across a recursive call was clobbered. Our fix (globals_end
boundary at the first function body + per-call frame restore in the
interpreter) is the architectural half.

What the paper suggests for the *second* half (once recursion runs in the JIT
again): treat **call boundaries as live-range terminators** and split intervals
there, instead of running a global interference graph. Concretely:
- an interval that crosses a call gets spilled at the call site and reloaded
  after it;
- values that die inside the callee never enter the caller's interval set;
- the allocator only needs the caller-saved set, not a whole-program graph.

That is a hypothesis to test against our allocator, not a proven port. The
Ackermann edge (nested recursive call as an argument) is exactly the case where
"split at the call" still has to keep the inner call's result live.

### COMPILER #38 — gauntlet valid_pass_rate > 70 %

No single paper moves this. The lever remains #37: every recursive program that
currently bails to the interpreter oracle is a program the JIT could own if
frame reconstruction is correct. Qwen3-Coder-Next's result (`2603.00729`) is
indirect evidence that **executable-environment feedback** is the strongest
training signal for code — which is what the gauntlet already is. The honest
path is: fix #37, re-run the 2,668-case gauntlet, and only then claim a rate.

### COMPILER #52 — LTO (cross-TU inlining)

Verified anchors:
- `1010.2196` — GCC's **WHOPR** design: parallel LTO, partitioning, summary-based
  cross-module optimization. This is the reference architecture for a
  whole-program pass that does not need one giant process.
- `1403.6997` — practical notes on optimizing *large* applications with GCC/LLVM
  LTO (code size, cold start, memory).
- `1807.06735` — **BOLT**: post-link binary layout optimization. Useful even
  without changing our front end: it shows layout alone is worth measurable
  throughput.

Action: a WHOPR-shaped plan is the right shape for WuBuNOS (partitioned
summaries → per-partition inlining → parallel codegen) rather than a monolithic
whole-program IR.

### COMPILER #55 — computed goto / labels-as-values

No dedicated paper survived verification for `GNU &&label` dispatch; the
practitioner evidence is in the blog layer (`layer6_blogs.jsonl`, e.g.
Brendan-Gregg-style dispatch notes, CPython/V8 JIT posts). Treat this as an
engineering item, not a research item:
- `#if defined(__GNUC__)` → `&&label` table;
- fallback → `switch` loop with the same opcode enum;
- measure both on the gauntlet, because branch-predictor behaviour differs
  sharply between threaded and switch dispatch on modern cores.

### KERNEL #13 — preemption p99 < 100 µs (currently env-blocked on WSL)

This is the strongest new lead in the wave. Verified:
- `2609.40141` **LLTA** — a simplicity-oriented open-source **WCET analyser**
  for COTS microcontrollers; the point is deriving a *safe upper bound* for the
  binary you actually flash, not a synthetic benchmark.
- `2604.27862` **AnTi-MiCS** — analytical framework for bounding time in
  mixed-criticality embedded systems.
- `2609.01077` **JENGA** — RowHammer countermeasures can *break real-time
  predictability*; i.e. worst-case latency is a system property, not just a
  scheduler property.

Action: stop trying to measure WSL latency. Build the cyclictest-style harness
**inside the guest** (WuBuOS on QEMU, and bare metal when available), record a
histogram, and report p50/p99/p99.9 with the same methodology LLTA/AnTi-MiCS
use for bounds. WSL numbers should never be cited as kernel latency.

### BRAIN #115–123 — the 8 Lean proofs

Verified:
- `2610.00760` **LeanSide** — co-reasoning system for natural-language proofs
  with machine-checked verification.
- `2610.20573` **Beyond LLM Judge** — combines LLM semantic review with
  **mechanical numerical and vacuity checks**; the key idea is that a proof can
  compile and still be vacuous, so compilation alone is not faithfulness.
- `2610.18498` — theory-scale autoformalization faithfulness.
- `2609.32908` — JEPA representations for Lean theorem proving.
- `2602.23981` **Intrinsic Lorentz Neural Network** and `2609.28248`
  **hyperbolix** (JAX) — concrete hyperbolic-layer implementations to compare
  against our Möbius/gyration C11 port.

Action for the 8 proofs (blocked on read-only `wubumath`): each proof must ship
with a **vacuity check** (the statement is not trivially true) and a numerical
spot-check of the C11 implementation against the Lean theorem. That is the
pattern the 2026 literature converged on, and it matches our Triple-DA.

### Architecture / OS — where our design is ahead or behind

- `2607.25076` **Towards an Agent Operating System** argues the field is at the
  "POSIX/Kubernetes moment": stable abstractions, portability, semantics.
  Our 9P/Styx namespace + capability object manager is exactly that move; the
  paper is a good external citation for the design, not a new mechanism.
- `2606.00279` **Bit-Exact AI Inference Verification** is a direct challenge to
  our test philosophy: cosine similarity is not verification. Where we can make
  a kernel bit-exact (integer dequant, Q8 paths), we should assert bit-exact
  equality against the oracle, and reserve tolerance tests for genuinely
  floating-point reductions.

---

## 3. Convergence map (what the 8 layers agree on)

1. **Bytes moved per token is still the spine.** 119 of 432 verified papers are
   inference-efficiency work; the KV/spec/quant cluster dominates.
2. **Sparse activation + a router is the default architecture.** Our router slot
   is the right abstraction; the open question is routing quality, not presence.
3. **Verification is splitting into two camps**: bit-exact where possible,
   mechanical vacuity/faithfulness checks where not. Both are stricter than
   cosine tests.
4. **Worst-case latency is a system property** (JENGA), so KERNEL #13 must be
   measured in-guest with a documented harness.
5. **Encoder-free / unified input is shipping** (Gemma 4 12B), which validates
   the single-encoder KV-FS direction.

## 4. What I did NOT find (honest gaps)

- No verified paper dedicated to **computed-goto dispatch**; #55 stays an
  engineering task.
- No verified paper on **recursive JIT frame reconstruction** specifically; our
  globals_end fix remains novel relative to the search, and the regalloc
  literature only informs the spill policy.
- No verified paper that gives a **>70 % gauntlet rate** recipe; #38 is ours to
  earn.
- Two PDFs failed to download (`2601.17473`, `2510.17898`); metadata exists,
  full text does not.

## 5. Next actions, ranked

1. Re-run the gauntlet after #37 lands; publish the real rate (no estimates).
2. Draft the WHOPR-shaped LTO plan for #52 from `1010.2196`.
3. Build the in-guest latency harness for #13 (LLTA-style histogram + bounds).
4. Add vacuity + numerical cross-checks to the Lean proof pipeline (#115–123)
   when `wubumath` is writable or via citation-only proofs.
5. Promote the 5–10 strongest new inference papers from the archive into
   `research/NNN-*.md` design docs after Triple-DA.
