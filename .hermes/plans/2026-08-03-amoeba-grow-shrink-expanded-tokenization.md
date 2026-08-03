# The Amoeba Full-Body Plan: all parts grow AND shrink + expanded tokenization for simpler depth

> **For Hermes:** Use subagent-driven-development to implement this plan task-by-task.

**Goal:** Turn WuBu from a one-directional grower (layers only) into a true amoeba — every part (layers, width, FFN, heads, vocab, selectors, patch stream) can grow AND shrink hive-style — and research+implement expanded (byte/patch) tokenization so the model reaches the same quality with SIMPLER depth.

**Architecture:** Two coupled upgrades on the existing seed (wubuwizard, C11, 35M, dim 448, 12 layers, byte-level BPE vocab 16384):
1. **Symmetric morphing operators** — the inverse of every existing grow op (zero-insert, stack) plus prune/merge ops (ShortGPT BI-score, LaCo merge), with the train state following each op (reverse SHIFT_ARR), all behind the amoeba fitness gate.
2. **BLT-style expanded tokenization** — bytes → dynamic entropy-patched → local encoder → shallower global transformer → local decoder, so depth is *simpler* (fewer global layers do the same work because the byte modules offload fine-grained structure).

**Tech Stack:** C11, own kernels only (no third party), safetensors loader, Muon+AdamW trainer, wubu_hive (skipfield+freelist), wubu_amoeba (diagnose/grow/shrink/validate), wubu_plateau, ASan/UBSan test gates. Research spine: BLT arXiv:2412.09871, ShortGPT arXiv:2403.03853, LaCo arXiv:2402.11187, Net2Net (Chen 2015), Gloeckle multi-token 2404.19737, BPE-dropout ACL2020.

---

# PHASE 0 — Audit (current state, what exists)

**Already there:**
- `src/wubu_grow.c` + `include/wubu_grow.h`: `wubu_grow_insert_block` (zero-insert, function-preserving), `wubu_grow_stack_block` (G_stack copy), `wubu_grow_schedule` (Zhiqi Bu monotonic), `wubu_train_grow` (shift grad/mom/norm pointers, SHIFT_ARR). **No shrink exists.**
- `src/wubu_amoeba.c`: diagnose → grow/shrink/stasis at the EXPERT (cell) level via `wubu_hive`; fitness gate (loss tol + prover + floor/ceiling). Its shrink is hive-slot-level, not model-part-level.
- `src/wubu_hive.c`: skipfield + freelist, O(1) insert/erase, stable pointers.
- `src/wubu_backprop.c`: Muon + AdamW real backward; `wubu_train_t` holds per-block grad/mom arrays (all 12 slots pre-allocated).
- Tokenizer: `src/wubu_tokenizer_hf.c` (byte-level BPE, vocab 16384, merges table) — BL08 wired.
- `wubu_model_t`: `blocks[12]`, `is_full[]`, `fire_sel[]`, `n_layers` (active count), selectors[3].

**The asymmetry to kill:** `wubu_grow_insert_block` shifts blocks up + `wubu_train_grow` shifts train arrays up. There is NO `wubu_shrink_*` and NO `wubu_train_shrink`. The amoeba module can't actually shrink the model body — it only recycles expert slots. Depth currently only goes up (420-step runs: 2→9 layers).

---

# PHASE 1 — Symmetric operators: every part grows AND shrinks

Design doctrine (the DA oracle for every op):
- **GROW** must be *function-preserving* (zero-init / Net2Net copy): forward-before == forward-after (verify bitwise, tolerance 1e-6).
- **SHRINK** is NOT function-preserving by definition — its oracle is the *amoeba fitness gate* (held-out loss within `loss_tol`, prover passes, BI-score-informed layer choice) + the FD backward check.
- **Train state follows**: every model op has a matching `wubu_train_*` op (reverse SHIFT_ARR for shrink: free the removed block's grad/mom/norm buffers, shift down, zero the tail slot).
- All block slots live in `m->blocks[]` as hive-ish slots: grow = freelist pop, shrink = freelist push (memory returns to the pool — the amoeba's membrane).

### Task 1.1 — `wubu_shrink_remove_block` (layer removal, ShortGPT-informed)

**Objective:** Remove the block at position `pos`; shift blocks [pos+1..n) down; free the removed block's buffers; keep rhythms consistent.

**Files:**
- Modify: `src/wubu_grow.c` (add), `include/wubu_grow.h`
- Test: `tools/test_grow.c` (extend: grow 2→5 then shrink 5→2, verify live count + forward sanity)

**Step 1: Write the failing test** (in `tools/test_grow.c`): grow to 5, record forward logits, `wubu_shrink_remove_block(&m, 2)`, assert `m.n_layers == 4`, assert blocks[2..3] == old blocks[3..4] (pointer equality), run forward → finite, no crash.

**Step 2: Run → FAIL** (`make test_grow`) — symbol missing.

**Step 3: Implement:**
```c
/* Remove the block at pos: shift [pos+1..n) down by one, free the
 * removed block's buffers, decrement n_layers.  NOT function-preserving;
 * the caller (amoeba) must validate via the fitness gate. */
int wubu_shrink_remove_block(wubu_model_t *m, int pos)
{
    if (!m || pos < 0 || pos >= m->n_layers) return 0;
    if (m->n_layers <= 1) return 0;
    block_free(&m->blocks[pos]);                 /* existing helper */
    for (int l = pos; l < m->n_layers - 1; l++) {
        m->blocks[l] = m->blocks[l + 1];         /* struct copy (move) */
        m->is_full[l] = m->is_full[l + 1];
        m->fire_sel[l] = m->fire_sel[l + 1];
        memset(&m->blocks[l + 1], 0, sizeof(wubu_block_t)); /* ownership transfer */
    }
    m->n_layers--;
    return 1;
}
```

**Step 4: Run → PASS.** **Step 5: Commit** `feat: wubu_shrink_remove_block (layer removal, amoeba apoptosis)`.

### Task 1.2 — `wubu_train_shrink` (train state follows the removal)

**Objective:** Reverse of `wubu_train_grow`: free the removed block's grad/mom/norm buffers, shift [pos..n) down, zero the vacated tail.

**Files:** `src/wubu_grow.c`, `include/wubu_grow.h`, `tools/test_grow.c`

**Step 1: failing test** — after grow+shrink pairs, `wubu_train_shrink(&tr, pos, n_layers)` returns 1; free of the train state is ASan-clean; per-block pointers for the shifted blocks match the model's shifted blocks (same ordering).

**Step 2: FAIL → Step 3: implement** (mirror of SHIFT_ARR, reversed):
```c
#define SHRINK_ARR(ARR) do {                                       \
        if (ARR[pos]) free(ARR[pos]);                               \
        for (int l = pos; l < n_layers - 1; l++) ARR[l] = ARR[l+1]; \
        ARR[n_layers - 1] = NULL;                                   \
    } while (0)
```
plus the norm-slot reverse (`norm_g[4l+k] = norm_g[4(l+1)+k]`, free `4*pos..4*pos+3`, zero the tail `4*(n_layers-1)..+3`).

**Step 4: PASS (ASan clean). Step 5: Commit.**

### Task 1.3 — BI-score oracle (`wubu_block_importance`)

**Objective:** ShortGPT's Block Importance: `BI(l) = E[‖x_{l+1}‖ − ‖x_l‖]` over a calibration batch — which layer changes the hidden state the least (most redundant). This is the *diagnose* signal for depth shrink.

**Files:** new `src/wubu_bi.c` + `include/wubu_bi.h`, test `tools/test_bi.c`

**Step 1: failing test** — feed a 2-layer model with one identity-ish block; assert the identity block scores ~0 (most redundant) and the real block scores >> 0.

**Step 3: implement** — reuse `wubu_bp_forward` with a hook, or a dedicated pass capturing `attn_norm` inputs per layer (norm pre-activations = the hidden states between layers); BI = mean over tokens of ‖h_{l+1}‖ − ‖h_l‖ (sign conventions per ShortGPT: low = redundant).

**Step 4: PASS. Step 5: Commit.**

### Task 1.4 — `wubu_shrink_merge_block` (LaCo layer merge)

**Objective:** Merge the two most-similar adjacent layers (deep→shallow, as LaCo does) instead of hard removal: new block = weighted mean of the pair (or collapse the pair into one). Used when BI says both are low but hard removal would break the residual stream.

**Files:** `src/wubu_grow.c`, `include/wubu_grow.h`, `tools/test_grow.c`
**Step 3 implement** — average the two blocks' buffers into the lower position, free the upper, shift down. **PASS + ASan + commit.**

### Task 1.5 — width grow/shrink (dim 448 ⇄ dim 512, Net2Net + pruning)

**Objective:** Grow `dim` by Net2Net (duplicate rows of q/k/v/o/g_proj, gate_up/down, embedding, norms — with the 1/2 replication factor on fan-in rows), function-preserving; shrink by low-norm column pruning (rows of the output matrices + corresponding embedding dims).

**Files:** new `src/wubu_width.c` + `include/wubu_width.h` (there is already a `tools/test_width.c` and `src/wubu_width.c` stub — extend), `src/wubu_train.c` (train arrays must realloc), test `tools/test_width.c`
**Note:** this touches every buffer size — the model is currently hard-`#define`-dim. Introduce `m->dim` runtime field with default BARUN_DIM and thread it through forward/backward (largest task in the phase; do it as its own sub-phase with the existing `tools/test_width.c` as the guard).

### Task 1.6 — FFN dim grow/shrink (1228 ⇄ bigger/smaller)

Same Net2Net/prune machinery restricted to `gate_up`/`down` (+ `ffn_gate`/`ffn_up` buffers). Test in `tools/test_width.c`.

### Task 1.7 — vocab grow/shrink (16384 ⇄ ±2048)

**Objective:** grow = add embedding rows (zero or copied from a shared tokenizer), shrink = prune rows for tokens with ~zero usage (corpus-count-driven). Tied embeddings: `embedding` and `lm_head` must move together.

**Files:** `src/wubu_tokenizer_hf.c` (vocab table is read-only — add a mutable extension table), `src/wubu_width.c`, `tools/test_width.c`

### Task 1.8 — selectors grow/shrink (3 ⇄ 2..4)

Selector array is `selectors[BARUN_SELECTORS]`; grow adds a zero score-vector, shrink drops the least-used. Minor.

### Task 1.9 — hive-back the block slots (the membrane)

**Objective:** `m->blocks[]` becomes hive-backed: grow pops the freelist (or allocates), shrink pushes. Same API as `wubu_hive` but for `wubu_block_t`. This is what makes shrink *cheap* and *recyclable* — the amoeba's membrane. Keep the existing shift semantics for the active window; the hive only manages the *inactive tail* slots so memory returns to the pool instead of being freed/leaked.

**Files:** `src/wubu_grow.c`, `include/wubu_grow.h`, extend `src/wubu_hive.c` with a typed block-slot variant (or keep a `wubu_block_t freelist[12]` + count — simpler, same property).

---

# PHASE 2 — Expanded tokenization: bytes → patches → simpler depth

Research synthesis (what the literature says):
- **BLT (2412.09871)**: byte-level, tokenizer-free; a *tiny* byte-LM (or 2-byte-context CNN) predicts next-byte entropy → bytes are grouped into *patches* of varying size (high-entropy = short patches = more compute; low-entropy = long patches = cheap). Three modules: **local encoder** (bytes→patch reps; hash n-gram embeddings, pooling, cross-attn to the global), **global latent transformer** (the main body — runs ONCE PER PATCH, not per byte), **local decoder** (patch reps→bytes). Result: matches Llama-3 flop-controlled with up to 50% fewer inference flops; robustness to noise/long-tail ↑.
- **The "simpler depth" mechanism**: the global transformer runs on PATCHES — for the same text it executes ~4–8× FEWER steps than a byte model and roughly the same as a BPE model, while keeping byte-level access. So the global body can be SHALLOWER (or the same depth with far less compute). Depth is no longer the only lever — **patch size is a grow/shrink axis** (BLT's new scaling axis).
- **Multi-token prediction (2404.19737)**: train the head to predict the next k tokens/patches in parallel — better sample efficiency for small models (the 35M seed is exactly the regime where MTP shines).
- **BPE-dropout (ACL2020)**: stochastic merge application during training → robust embeddings, better generalization; trivial to add to the existing BPE tokenizer.

### Task 2.1 — the entropy model + patcher (`wubu_patch`)

**Objective:** a tiny byte-LM (2 layers, dim 64, context 2–8 bytes — or the BLT "small CNN byte model with 2-byte context" baseline) that scores next-byte entropy; a patcher that walks a byte stream and emits patch boundaries where entropy > threshold (with the monotonicity + newline-reset constraints from §4.4 of the paper). Fixed-size and entropy variants both supported (stride-4 for tests, entropy for real).

**Files:** new `src/wubu_patch.c` + `include/wubu_patch.h`, test `tools/test_patch.c`
**Oracles:** (a) patcher round-trip: bytes → patches → bytes == identity; (b) entropy model trained on the corpus reaches next-byte cross-entropy below the BPE tokenizer's byte-equivalent CE; (c) on repetitive text, avg patch size ↑ (compute savings real).

### Task 2.2 — the local encoder (bytes → patch representations)

**Objective:** BLT §3.2: per-byte embeddings + **hash n-gram embeddings** (no learned vocab for patches!) + pooling to one vector per patch + a few encoder transformer layers + **cross-attention** from patch reps to byte reps (pre-LN, no positional embeddings, masked to the patch's own bytes).

**Files:** new `src/wubu_blt.c` (encoder half) + `include/wubu_blt.h`, test `tools/test_blt.c`
**Oracle:** patch rep for "Mo" in "Mozart" ≈ informative — FD-check the encoder's gradient flow; encoder output dim matches the global model's dim.

### Task 2.3 — the local decoder (patch reps → bytes)

**Objective:** BLT §3.3: alternating cross-attention (byte queries ← patch key/values) + decoder transformer layers; predicts the next byte distribution. Tied to the existing vocab head for the byte vocabulary (256 + specials).

**Files:** `src/wubu_blt.c` (decoder half), test `tools/test_blt.c`
**Oracle:** end-to-end byte LM loss on a 1M-byte slice; beats the BPE path on character-level tasks (spelling, noise robustness — the BLT selling points).

### Task 2.4 — multi-token/byte prediction head

**Objective:** Gloeckle's MTP: k=2–4 parallel prediction heads on the last global layer (shared trunk, per-depth heads). Auxiliary loss at 0.3 weight.

**Files:** `src/wubu_backprop.c` (add heads + loss), `include/wubu_train.h`, test in `tools/test_backprop.c`
**Oracle:** same-token-budget comparison: MTP model reaches the held-out loss floor faster (sample efficiency).

### Task 2.5 — BPE-dropout on the existing tokenizer

**Objective:** in `wubu_tokenizer_hf`, with prob `p` skip each merge during encode (train only). Zero cost, robustness win, keeps the 16384 vocab.

**Files:** `src/wubu_tokenizer_hf.c`, test `tools/test_wubu_save.c` (round-trip must still hold — dropout only during training encode, never in the saver)

### Task 2.6 — integrate with the amoeba (patch size is a grow/shrink axis)

**Objective:** the entropy threshold is a *morphable parameter*: when the model is overworked (high util), lower the threshold → more, shorter patches → more compute; when idle, raise it → long patches → cheaper. This is the hive/floodgate of the body: the model grows and shrinks its *compute per token* without changing a single weight.

**Files:** `src/wubu_amoeba.c` (add patch-axis to diagnose/mutate), `src/wubu_patch.c` (threshold setter)

---

# PHASE 3 — The hive-fast path (make it fast)

Design: the patch stream is a hive of cells (skipfield marks patch boundaries; freelist recycles released patches). The entropy model runs once per byte with a 2–8 byte window (O(1) amortized, SIMD-friendly). Hash n-gram embeddings are a single table lookup (no vocab training, no OOV). The global transformer is the only expensive module and it runs `bytes/avg_patch` times — that ratio is the whole speed story.

**Task 3.1** — hive-backed patch stream (`wubu_patch_stream`): patches allocated from a block pool; erase/recycle O(1); foreach jumps skips. Reuse `wubu_hive` semantics.
**Task 3.2** — benchmark gate: `tools/bench_patch.c` — bytes/sec through the full pipeline (entropy → patch → encoder → 6-layer global → decoder) vs the current 12-layer BPE path. Target: same quality, ≤60% of the flops, decode latency ≤ current.
**Task 3.3** — end-to-end train test: train the patched 6-layer body on the corpus (finemath-live.tok), verify loss trajectory beats the 12-layer BPE baseline at the same wall-clock.

---

# Tests / validation (the DA matrix)

| Op | Oracle |
|---|---|
| grow insert/stack | forward-before == forward-after (1e-6), FD backward |
| shrink remove/merge | amoeba fitness gate (loss tol 0.05) + BI-informed choice + FD backward |
| train grow/shrink | ASan-clean free, pointer ordering matches model |
| patcher | bytes→patches→bytes identity; avg patch size ↑ on repetitive text |
| encoder/decoder | CE on 1M bytes; FD gradient flow |
| MTP | sample-efficiency comparison at fixed token budget |
| whole system | `make test_grow test_amoeba test_blt test_patch test_width test_backprop` all green; full 420-step run EXIT=0 ASan-clean |

Every task: **write failing test → run to see it fail → implement → run to see it pass → commit.** No stubs.

# Files that will change

- `src/wubu_grow.c` / `include/wubu_grow.h` — shrink ops + train_shrink + BI hook + hive-backed slots
- `src/wubu_bi.c` / `include/wubu_bi.h` — NEW, ShortGPT BI oracle
- `src/wubu_width.c` / `include/wubu_width.h` — width/FFN/vocab morphing (runtime `m->dim`)
- `src/wubu_patch.c` / `include/wubu_patch.h` — NEW, entropy model + patcher
- `src/wubu_blt.c` / `include/wubu_blt.h` — NEW, local encoder + decoder
- `src/wubu_backprop.c` — MTP heads, dim-runtime threading
- `src/wubu_tokenizer_hf.c` — BPE-dropout, mutable vocab extension
- `src/wubu_amoeba.c` — patch-size axis, model-part shrink in diagnose/mutate
- `src/wubu_train.c` — dim-runtime train arrays
- `tools/test_grow.c test_width.c test_backprop.c test_amoeba.c` + new `tools/test_bi.c test_patch.c test_blt.c bench_patch.c`

# Risks / tradeoffs / open questions

- **dim-runtime refactor is invasive** (every buffer is `#define`-sized today). Mitigate: keep BARUN_DIM as the default; introduce `m->dim` and thread it; `tools/test_width.c` already exists as a guard. This is the highest-risk task — do it first within Phase 1.5, in isolation.
- **Shrink ≠ function-preserving**: a bad removal must roll back (5+1 recovery exists on the metal side; wire the same for the hosted trainer). BI score + loss-tol gate is the safety.
- **BLT cost at 35M scale**: BLT's wins are clearest at 1B+; at 35M the byte modules may eat the savings. The honest experiment is Task 3.3's wall-clock comparison — if the patched 6-layer doesn't beat the 12-layer BPE at equal wall-clock, keep the BPE path as the default and ship BLT as the wubu_mode=2 path.
- **Entropy model training budget**: the tiny byte-LM needs corpus time; reuse the existing finemath-live.tok stream (no new data).
- **Open question**: should vocab shrink be corpus-count-driven (offline) or usage-driven (online, via the amoeba)? Start offline (deterministic), flip online later.

# Execution order

Phase 0 audit (done above) → 1.5 dim-runtime FIRST → 1.1/1.2/1.3/1.4/1.9 (depth shrink) → 1.6/1.7/1.8 (width/FFN/vocab/selectors) → Phase 2.1/2.2/2.3 (BLT core) → 2.4/2.5 (MTP + BPE-dropout) → 2.6 (amoeba integration) → Phase 3 (hive-fast + benchmarks).
