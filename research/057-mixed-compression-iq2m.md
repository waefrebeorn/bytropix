# research/057 — MIXED COMPRESSION: the IQ2_M / Unsloth 7-hop (quality density)

> 2026-08-04. The user's directive: "research the mixed iq2m unsloth seven
> steps to Kevin Bacon use that... we should be generalized using the
> greatest compression ever known demand which is mixed compression that
> allows us to keep maximum elements of what we need and then minimize
> what we need less. Go for it — accuracy crap."

## The anchor: OUR artifact dissected

`Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (12 GB, Unsloth Dynamic IQ2_M) — opened
with the tensor store, 753 tensors. The REAL Unsloth UD mixed plan:

| Role | Quant | bpw | elems | the doctrine |
|---|---|---|---|---|
| expert gate/up (the bulk) | **IQ3_XXS** | ~3.06 | 20.9 B | the saturated low-rank — MINIMIZE |
| expert down | **IQ4_NL** | ~4.5 | 9.9 B | the expansion direction — more bits |
| shared-expert down | **IQ2_XXS** | ~2.06 | 0.3 B | heavily-used — most aggressive |
| attention QKV + gates | **Q8_K** | ~8.5 | 1.6 B | attention is sensitive — KEEP MAX |
| token_embd | Q8_K | ~8.5 | — | the semantic root — KEEP MAX |
| output (lm_head) | Q6_K | ~6.6 | 0.5 B | |
| norms / routers (368) | F32 | 32 | tiny | EXACT |
| strays | Q2_K/Q4_K/Q5_K/IQ1_M/BF16 | — | — | Unsloth's per-tensor sweep |

**The one-line convergence: compression is a LADDER over roles, never a
uniform bit-width.** Keep maximum elements where signal lives (attention,
embeddings, norms); minimize where saturation eats the bits (expert
weights). 12.3 GB of 35B with HermesAgent-20 #1 (Escha) is the proof.

## The 7 hops (Kevin Bacon, mixed-quant lineage)

1. **IQ2_M** — ggml's 2-bit "medium": per-BLOCK mixed 2/3-bit + 8-bit
   super-scales + importance-matrix (activation-order) grids. The "M"
   means the block itself is mixed. The family: IQ1_S (1.56) → IQ1_M →
   IQ2_XXS (2.06) → IQ2_XS → IQ2_S → IQ2_M (2.4) → IQ3_XXS (3.06) →
   IQ3_S (3.44) → IQ4_NL (4.5) → IQ4_XS (4.25).
2. **The importance matrix** — quants are fit to the ACTIVATION order,
   not raw weight order (act-order: row-wise importance reorders the
   grid search). This is why 2-bit IQ beats naive 2-bit: the bits go
   where the activations flow.
3. **The k-quants** — Q2_K..Q8_K: mixed WITHIN the block (2-6-bit pieces +
   8-bit scales). The predecessor; the IQ series is its
   importance-matrix refinement.
4. **Unsloth UD (Unsloth Dynamic)** — the per-TENSOR allocation across
   the IQ family, automated by sensitivity. Our 35B-A3B is the artifact:
   gate/up→IQ3_XXS, down→IQ4_NL, shared→IQ2_XXS, attn/emb→Q8_K,
   head→Q6_K, norms→F32. The "mixed compression" the user names.
5. **Escha W2** (llm.ciru.ai/research/escha-vs-35b, research/046 AM03) —
   2b gate/up · 3b down · INT8 dense = 12.3 GB, HermesAgent-20 90/100
   (#1 of 12), within 2.4pp of the best HumanEval+. QUALITY DENSITY:
   per-family bit-width chosen by sensitivity beats any uniform quant.
6. **DeepSeek Config-I** (the KAHUNA, 2.88 bpw effective) — Q2_0 expert
   gate/up + TQ3_1S (WHT-rotated 3-bit) + q8_0 token_embd + q6_K head +
   f32/bf16 norms/routers. The production mixed recipe at 284B scale —
   the same ladder shape, verified.
7. **Our engine** — the tensor store (`wubu_tensor_store`, AN07) already
   reads every format; the mixed EXPORT (`wubu_ts_export_mixed`, this
   wave) applies the ladder to ANY model we hold. Our reader dequantizes
   the whole IQ family (IQ1_S/M, IQ2_XXS/S, IQ3_XXS/S, IQ4_XS, Q8_K,
   Q6_K, Q4_K, Q5_K in-tree) — the encoders are what we now own.

## The generalized doctrine (the implementation ladder)

Per-role bit assignment for ANY model (the plan structure):

| Role (name classifier) | Quant | why |
|---|---|---|
| `embed` / `token_embd` | Q8_K/Q8_0 | the semantic root; ~21% of 35M params — keep max |
| `head` / `lm_head` / `output` | Q6_K/Q8_0 | tied to embed — high |
| `attn` (q/k/v/o, qkv) | Q8_K/Q8_0 | attention is sensitive — keep max |
| `gate_inp` / router / `norm` / tiny | F32 | exact, negligible bytes |
| expert `gate`/`up` | IQ3_XXS (or IQ2_XXS) | saturated low-rank — minimize |
| expert `down` | IQ4_NL | expansion — more bits |
| shared expert | IQ2_XXS | the aggressive end |

"Keep maximum elements of what we need, minimize what we need less."
Accuracy is a constraint, not the goal — the goal is quality DENSITY.

## Implementation (this wave + next)

- `wubu_ts_export_mixed(ts, out, plan)`: per-role quant assignment over
  the tensor catalog; streaming (one tensor at a time). Working encoders:
  F32, Q8_0 (verified byte-correct vs the reader), Q4_0 (fp16 d + nibbles).
- Next: IQ2_XXS + IQ3_XXS + IQ4_NL encoders (grids + super-scales +
  importance matrices — the dequant + grids are in-tree:
  `dequant_iq2_xxs.c`, `iq2xxs_grid_data.inc`, etc.; the encoders mirror
  them). Then the full Unsloth ladder applies to the seed, the zoo, and
  the KAHUNA when it lands.
- The Escha/AM03 precision-plan module (`wubu_precision_plan`) consumes
  the same ladder — one source of truth for per-family bits.

## Registration

- INDEX theme AN entry AN09 (this doc) + AN07's tensor store extended.
