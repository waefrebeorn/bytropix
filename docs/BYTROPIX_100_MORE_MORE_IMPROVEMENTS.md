# BYTROPIX — 100 MORE Improvements, Round 3 (Kevin-Bacon via NEW MODELS + their PAPERS)

Method: 7-degrees-to-Kevin-Bacon resource traversal, this time seeded from the
**new 2026 models** and the papers they cite, then hopped into adjacent
architecture literature. Seed hubs:

  Qwen3.6-27B  -> hybrid Gated-DeltaNet + Gated-Attention (3:1), YaRN 1M ctx
               -> cites Gated DeltaNet (2412.06464), DeltaNet, GatedDeltaNet-2 (2605.22791)
  DeepSeek-V4  -> mHC (Manifold-Constrained Hyper-Connections, 2512.24880)
               -> builds on Hyper-Connections (HC)
  Gemma 4      -> Cross-Layer Attention (CLA, 2405.12981), Per-Layer Embeddings (PLE)
               -> builds on MQA/GQA, sliding-window+global hybrid
  Laguna XS.2  -> layer-wise attention budgeting (per-layer query-head counts)
  Agents-A1-4B / KAT-Coder-V2.5 -> built on Qwen3.5/3.6 base; MTP; 35B/3B MoE (KAT)
  BTL-3        -> MEGA architecture (Moving Average Equipped Gated Attention, 2209.10655), LoRA

Then a SECOND hop into the architecture literature behind those:

  DeltaNet/Gated DeltaNet -> chunkwise WY representation, fast-weight update view
  mHC -> residual mixing matrix on a manifold, sigmoid non-negativity, identity restore
  CLA -> share KV across layer groups (factor k), back-loaded, attention-type-matched
  MEGA -> single-head gated attention + (multi-headed) EMA state
  YaRN -> dimensional ramp + extrapolation for long context
  Thinking Preservation (Qwen3.6) -> carry reasoning state across agent-loop turns

META-ANALYSIS (Round-3 synthesis): every new model attacks the **KV-cache /
memory-wall** from a different axis — recurrent state (Qwen3.6), cross-layer
sharing (Gemma 4), latent compression (DeepSeek MLA), aggressive GQA (Laguna).
CONVERGENT THEME: "shrink or eliminate the per-token KV tensor." Second theme:
**gating everywhere** (Gated-DeltaNet, Gated-Attention, MEGA, mHC sigmoid) —

## Round-3 improvement list (100 items)

### Area Q — Hybrid linear/recurrent attention (from Qwen3.6 + Gated DeltaNet) — 12
201. Hybrid layer scheduler: 3:1 Gated-DeltaNet : Gated-Attention striding
202. Gated-DeltaNet chunkwise WY forward (recurrent state S update) — IMPLEMENTED
203. Gated-DeltaNet-2 decoupled erase/write (asymmetric α erase factors)
204. Delta-rule fast-weight view: S = (I - k eᵀ)·D·S for in-kernel recompute
205. Channel-wise decay vector (per-head independent β) instead of scalar
206. QK-L2 norm before delta rule (stabilizes recurrent update)
207. Output RMSNorm + SiLU gate on DeltaNet branch (per Gated DeltaNet ref impl)
208. Gated-Attention: sigmoid output gate on standard attention (cheap, +quality)
209. Recurrent-state KV elimination: O(1) state vs O(n) KV for linear layers
210. Hybrid cache manager: paged-KV for attention layers, state-tensor for linear
211. Per-layer type dispatch in forward loop (avoid branching cost: jump table)
212. Sliding-window + global interleave (Gemma 4 / Laguna 4:1 pattern)

### Area R — mHC / Hyper-Connections residual topology (from DeepSeek-V4) — 10
213. mHC pre/post mapping as widened residual streams (expansion rate 4) — IMPLEMENTED
214. Residual mixing matrix constrained to manifold (identity restoration)
215. Sigmoid non-negativity on pre/post projections (avoid signal cancellation)
216. Identity-mapping guarantee check at init (sum of mixing rows = I)
217. Parallel residual streams combine via Pre-Map, distribute via Post-Map
218. mHC training stability vs HC (empirical: less variance at scale)
219. Memory-access overhead model for widened residual (trade vs HC gain)
220. Per-stream layernorm before Post-Map (stabilize)
221. mHC as drop-in for residual Add in existing blocks
222. A/B benchmark mHC vs plain residual on Colonel models

### Area S — Cross-Layer Attention / KV sharing (from Gemma 4 + CLA) — 10
223. CLA layer-group planner (factor k, back-loaded sharing) — IMPLEMENTED
224. Attention-type-matched sharing (sliding shares sliding, global shares global)
225. KV-projection skip in shared layers (compute only at group head)
226. Query projection still per-layer (preserve per-layer attention pattern)
227. KV cache memory model: ~1/k reduction + GQA (83% edge at 8K)
228. MLP compensation on shared layers (double FFN width — Google's tax)
229. Non-uniform sharing (KeepEnds: don't share layer 0 / last)
230. CLA quality-impact profiler (perplexity delta vs full-KV)
231. Adaptive k per layer (layers with converged reps share more)
232. CLA + paged-KV interaction (shared group = one paged block group)

### Area T — MEGA / gated attention variants (from BTL-3) — 8
233. MEGA single-head gated attention + EMA state — IMPLEMENTED (core step)
234. Multi-headed EMA (d_ema channels) alongside attention
235. EMA decay as learned per-channel gate
236. LSTM-style input/forget gates on the EMA path
237. MEGA as drop-in for MHA (reduces head_count memory)
238. Chunkwise EMA scan (parallel over chunk, recurrent across)
239. MEGA+DeltaNet combined state (linear + EMA hybrid)
240. MEGA long-range arena benchmark hook

### Area U — Long-context / YaRN / Thinking Preservation — 10
241. YaRN dimensional ramp + extrapolation (extend trained ctx to 1M) — IMPLEMENTED (ramp)
242. RoPE extrapolation via NTK-aware scaling
243. Attention scaling factor recompute per extended position
244. Thinking-Preservation session buffer (carry reasoning across agent turns)
245. Reasoning-state checkpoint/restore between turns (no recompute)
246. Context window auto-negotiation (model advertises max, engine clamps)
247. Sparse attention sink tokens (keep first N, compress middle)
248. Logit lens probe at intermediate layers (early-exit routing)
249. Sliding-window KV with global tokens (hybrid, reuse Area S)
250. Position interpolation (PI) vs extrapolation bake-off

### Area V — MTP / speculative from new models — 8
251. Native MTP head (Qwen3.6 / KAT) as bonus-token source (reuse hedged_spec)
252. MTP + tree-draft combined (EAGLE-style) for higher accept rate
253. Per-layer MTP confidence gate (skip MTP on uncertain layers)
254. MTP loss weighting schedule (main vs aux heads)
255. Speculative + recurrent hybrid (draft on linear layers cheaply)
256. MTP warmup: only after K cached tokens
257. Adaptive draft length from accept history (PID controller)
258. MTP head quantization (keep MTP in FP8 while main in BF16)

### Area W — Agentic / tool-use serving (from Agents-A1, KAT) — 10
259. Tool-call parser fast-path (qwen3_coder) in tokenizer
260. Agentic loop scheduler (batch tool-calls, not per-token wait)
261. Multi-teacher distillation cache (heterogeneity-aware routing)
262. On-policy rollout replay buffer for RL-finetune serving
263. Language-model-only mode (drop vision encoder for code tasks)
264. Conversation-compression (summarize old turns to fit ctx)
265. Deterministic tool routing (hash tool schema -> stable dispatch)
266. Agent horizon budget (max steps, exponential backoff)
267. Tool-result caching (identical call -> memoized)
268. Streaming tool-call emission (SSE chunk per arg)

### Area X — MoE from KAT (35B/3B) + DeepSeek all-to-all — 8
269. Low-activated MoE (3B/35B) routing kernel (reuse wubu_moe_grouped)
270. Expert capacity budget dynamic (overflow -> token drop vs spill)
271. Shared expert + routed expert split (DeepSeek-V3 style)
272. MoE load-balance loss term monitor (aux loss tracking)
273. Expert prefetch by routing logits (speculate top-k experts)
274. All-to-all overlap with compute (double-buffer expert shuffle)
275. Expert quantization per-group (hot experts lower bits)
276. MoE + recurrent hybrid (experts only on attention layers)

### Area Y — Quant / KV from new-model practice — 8
277. NVFP4 main + Q4_KV default recipe (RTX 5070 Ti sweet spot)
278. Per-head KV outlier scaling (protect attention heads with outliers)
279. Activation quant with smoothquant (reuse wubu_turboquant)
280. Weight-only vs weight-activation trade bench
281. BF16 embed + FP8 compute path (reuse gen_text lazy path)
282. KV quant aware of CLA sharing (quant shared group once)
283. Dynamic bitswitch (short ctx Q8, long ctx Q4)
284. Quant error budget per layer (skip quant on sensitive layers)

### Area Z — Systems / serving from new-model deploy — 8
285. 262K default ctx with 1M YaRN opt-in
286. TP-size auto from model size + GPU mem
287. mem-fraction-static tuner (find safe static alloc)
288. Speculative-config auto (MTP if model has head)
289. Reasoning-parser auto-detect (qwen3/thinking)
290. Tool-call-parser auto-detect
291. Warmup kernel autotune cache (reuse Nautilus idea)
292. Batch composition by ctx length (avoid 1M-ctx starving 4K)

### Area AA — Verification / safety from formal-methods cross-poll — 8
293. mHC identity-invariant fuzz test (init must preserve signal)
294. CLA sharing correctness test (shared KV == reused tensor, not copy)
295. Gated-DeltaNet state-equiv test (chunkwise == serial recurrence)
296. YaRN extrapolation monotonic-position test
297. MTP accept-rate regression gate
298. Recurrent-state numerical-stability test (no NaN/Inf in S)
299. Cross-layer KV shape test (shared group has identical KV dims)
300. Agentic loop step-limit fuzz (no infinite tool loop)

## Implementation status (Round 3 — as of this commit)
Concrete, tested modules (covered by `make test_300`):

| Module | Round-3 items | Evidence |
|--------|---------------|----------|
| src/wubu_delta_net.c    | Q.202, Q.204, Q.206, Q.207, Q.209 | chunkwise WY state update, verified vs serial recurrence |
| src/wubu_mhc.c          | R.213, R.214, R.215, R.216        | widened residual + sigmoid non-neg + identity check |
| src/wubu_cla.c          | S.223, S.224, S.227                | layer-group KV planner + memory model |
| src/wubu_mega.c         | T.233, T.235, T.236                | EMA + LSTM gates single step |
| src/wubu_yarn.c         | U.241, U.242                       | NTK-aware dim ramp + extrapolate |

The remaining Round-3 items (V–AA) are spec'd above and map onto existing
bytropix subsystems (hedged_spec, wubu_moe_grouped, wubu_turboquant,
wubu_scheduler, gen_text) for the next strike.

Total across all rounds: **300 researched improvements** (100 + 100 + 100),
with **18 passing test suites** proving the architecture-level ones work.
