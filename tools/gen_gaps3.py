#!/usr/bin/env python3
"""KB-7hop sweep part 3: Themes IX (security/fuzzing), IY (prompt
compression), IZ (MoE), JA (architecture hybrids), JB (multimodal),
JC (quantization). ~400 gaps."""
T = []

T.append("""
## Theme IX: Fuzzing / robustness / security
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (prompt-fuzzing evasion 2026; LogicFuzz NDSS 2026; autonomous fuzzing CERT 2026; EU-AI-Act robustness)
- IX01 Prompt-fuzz harness (adversarial prompt variants) `open`
- IX02 Evasion-rate measurement (per-category guardrail evasion) `open`
- IX03 Guardrail sensitivity matrix (keyword-adjacent robustness) `open`
- IX04 Autonomous fuzzing pipeline (LLM-supervised fuzzing) `open`
- IX05 Crash validator (filter unreachable crashes) `open`
- IX06 Fuzz-log analysis (LLM trace triage) `open`
- IX07 Semantic-fuzz oracle (behavior divergence, not just crashes) `open`
- IX08 Coverage-guided prompt mutation `open`
- IX09 Robustness regression gate (fuzz on every model change) `open`
- IX10 Adversarial-prompt taxonomy (jailbreak categories) `open`
- IX11 Robustness benchmark suite (measurable robustness) `open`
- IX12 Guardrail stress profile (per-guardrail weakness map) `open`
- IX13 Input-validation layer (schema-check adversarial input) `open`
- IX14 Fuzz-seed curation (high-value seed prompts) `open`
- IX15 Mutation operator library (prompt mutation ops) `open`
- IX16 Robustness scorecard (per-model robustness metrics) `open`
- IX17 Prompt-injection detector (injection-pattern classifier) `open`
- IX18 Output-validation gate (validate generated output) `open`
- IX19 Fuzz-round budget (bounded fuzz campaigns) `open`
- IX20 Vulnerability triage ledger (found + fixed registry) `open`
- IX21 Robustness-vs-quality tradeoff monitor `open`
- IX22 Adversarial example archive (replayable attack corpus) `open`
- IX23 Fuzzer self-healing (auto-recover fuzz stalls) `open`
- IX24 Robustness telemetry (per-input robustness signals) `open`
- IX25 Input-schema fuzzing (malformed structured input) `open`
- IX26 Injection-mitigation layers (defense-in-depth) `open`
- IX27 Robustness delta tracking (regression detection) `open`
- IX28 Fuzz coverage metrics (prompt-space coverage) `open`
- IX29 Adversarial distillation defense (robust training signal) `open`
- IX30 Fuzz-oracle calibration (false-positive control) `open`
- IX31 Prompt-leak detection (data-exfiltration guard) `open`
- IX32 Robustness energy budget (fuzz under J cap, ties IJ) `open`
- IX33 Security audit ledger (auditable security posture) `open`
- IX34 Input canonicalization (normalize adversarial variants) `open`
- IX35 Fuzz differential testing (same input, model variants) `open`
- IX36 Robustness auto-repair (detect + patch weak guardrails) `open`
- IX37 Adversarial robustness eval harness (NDSS-style) `open`
- IX38 Injection-resistance benchmark (standardized evals) `open`
- IX39 Fuzz-to-fix loop (fuzz finds, fix verifies) `open`
- IX40 Robustness model card (documented robustness) `open`
- IX41 Input-token anomaly detection (outlier input detection) `open`
- IX42 Guardrail redundancy (overlapping defenses) `open`
- IX43 Fuzz mutation seeds from real incidents `open`
- IX44 Robustness under resource limits (degraded-but-safe) `open`
- IX45 Security regression CI (fuzz in the pipeline) `open`
- IX46 Adversarial-prompt generation (auto-generate attacks) `open`
- IX47 Robustness attribution (which layer failed) `open`
- IX48 Fuzz-parallelization (parallel fuzz workers) `open`
- IX49 Injection-resistance training (robust fine-tune) `open`
- IX50 Robustness SLA (minimum robustness bar) `open`
- IX51 Fuzz campaign reports (structured findings) `open`
- IX52 Guardrail evolution (update guardrails from findings) `open`
- IX53 Adversarial robustness scoring (quantified defense) `open`
- IX54 Fuzz-verifier integration (fuzz feeds the verifier) `open`
- IX55 Robustness debt tracking (known weaknesses ledger) `open`
- IX56 Input-entropy guard (reject adversarial entropy spikes) `open`
- IX57 Robustness provenance (which defense caught what) `open`
- IX58 Fuzz coverage dashboards `open`
- IX59 Adversarial robustness transfer (attacks transfer across models) `open`
- IX60 Robustness-aware sampling (defense-aware decode) `open`
- IX61 Security-posture operator (auto-apply robustness configs) `open`
- IX62 Fuzz memory-safety (C-level crash fuzz, ties the kernel) `open`
- IX63 Robustness regression tests (per-gap assertion) `open`
- IX64 Adversarial input ledger (append-only attack log) `open`
- IX65 Robustness calibration (threshold fitting on attacks) `open`
- IX66 Security benchmark comparison (vs baseline defenses) `open`
- IX67 Fuzz-to-operator loop (findings drive config promotion) `open`
Status: `open` (67 gaps; fuzz/evasion measurement + autonomous fuzzing + guardrail hardening)
""")

T.append("""
## Theme IY: Prompt compression / context budgeting
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (LLMLingua-2; LongLLMLingua; RECOMP; Doc2Atom; Cartridges/CAS; LaMR; SES-RAG; GRC; EPC)
- IY01 LLMLingua perplexity-gated token drop (small-LM scoring) `open`
- IY02 LLMLingua-2 token classification (distilled BERT-level compressor) `open`
- IY03 LongLLMLingua question-aware reordering `open`
- IY04 Selective-Context self-information pruning (2x content, 40% compute) `open`
- IY05 RECOMP extractive+abstractive compression with selective augmentation `open`
- IY06 Doc2Atom compositional parametric memory (knowledge atoms + micro-LoRA) `open`
- IY07 Cartridges at Scale (modular KV caches, distractor mixing, budget manager) `open`
- IY08 LaMR multi-rubric code-context pruning (semantic + dependency CRFs) `open`
- IY09 SES-RAG semantic segmentation + query expansion + density truncation `open`
- IY10 GRC unified generation/retrieval/compression (meta latent tokens) `open`
- IY11 EPC expected-predictive compression (write-time retention by predicted questions) `open`
- IY12 Lost-in-the-middle mitigation (reorder important context) `open`
- IY13 Lexical-density-aware budgeting (dense contexts need more budget) `open`
- IY14 Tool-schema compression (44-50% schema token savings, ties agentic) `open`
- IY15 In-context autoencoder (continuous-embedding context) `open`
- IY16 Context distillation to LoRA (Doc-to-LoRA) `open`
- IY17 Latent-memory generation (compressed KV as updatable memory) `open`
- IY18 Hybrid paged attention for compressed context `open`
- IY19 Compression-ratio governor (target ratio with quality guard) `open`
- IY20 Compressed-prompt fidelity audit (reconstruction check) `open`
- IY21 Question-aware compression (query-conditioned retention) `open`
- IY22 Task-agnostic compressor (works across tasks) `open`
- IY23 Compression benchmark harness (compression quality evals) `open`
- IY24 Streaming compression (compress incrementally) `open`
- IY25 Compression energy accounting (compress vs not, ties IJ) `open`
- IY26 Retrieval-aware compression (retain retrieval-critical spans) `open`
- IY27 Per-token importance score caching `open`
- IY28 Compression curriculum (progressively harder compression) `open`
- IY29 Compressor-model choice (small-LM vs classifier vs heuristic) `open`
- IY30 Compression telemetry (ratio, quality, latency) `open`
- IY31 Context-budget planner (budget per stage: system/prompt/evidence) `open`
- IY32 Evidence-retention sufficiency (answerability check) `open`
- IY33 Compression + RAG integration (compress retrieved docs) `open`
- IY34 Agentic context pruning (multi-turn agent contexts, LaMR-style) `open`
- IY35 Compressed-KV paging (compressed pages) `open`
- IY36 Compositional compression (atom-level composition) `open`
- IY37 Compression provenance (what was compressed away) `open`
- IY38 Query-router for atom selection (Doc2Atom-style) `open`
- IY39 Micro-adapter injection (per-atom LoRA, ties lora) `open`
- IY40 Cartridge rotation (budget-managed cartridge swap) `open`
- IY41 Compression-quality monitor (post-compression performance) `open`
- IY42 Compress-or-keep decision (selective compression) `open`
- IY43 Token-budget inheritance (parent -> child agent budgets) `open`
- IY44 Compression-aware sampling (compressed context sampling) `open`
- IY45 Context-density profiler (density per window) `open`
- IY46 Compressed-prompt safety (never compress safety instructions) `open`
- IY47 Compression rollback (keep the original if quality drops) `open`
- IY48 Multi-stage compression (compress progressively) `open`
- IY49 Compression verification (answerability after compression) `open`
- IY50 Context-budget operator (auto-budget by task, ties IK) `open`
- IY51 Compressed-memory integration (compressed context as memory, ties IP) `open`
- IY52 Compression under energy budget (compress to save J) `open`
- IY53 Token-cost ledger (compression savings accounting) `open`
- IY54 Compression benchmark vs full-context baseline `open`
- IY55 Adaptive compression ratio (per-request ratio) `open`
- IY56 Compressor staleness (re-compress on context change) `open`
- IY57 Compression + eviction integration (compressed + evicted, ties IO) `open`
- IY58 Compressed-prompt telemetry (per-prompt stats) `open`
- IY59 Compression failure handling (fallback to full context) `open`
- IY60 Cross-model compression transfer (compress once, use anywhere) `open`
- IY61 Compression dataset distillation (train compressor from LLM outputs) `open`
- IY62 Context-budget fairness (per-tenant compression, ties IR) `open`
- IY63 Compression provenance audit (reproducible compression) `open`
- IY64 Compressed-context continual learning (compress + learn, ties BB) `open`
- IY65 Compression robustness (compressed adversarial input) `open`
- IY66 Compressor model portability (no external LM dependency) `open`
- IY67 Compression-to-operator loop (compression config promotion) `open`
Status: `open` (67 gaps; LLMLingua-family + RECOMP + cartridges + latent memory + density-aware budgeting)
""")

T.append("""
## Theme IZ: Mixture-of-experts routing frontier
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (Routing-Free MoE 2604.00801; PathMoE 2603.18297; expert specialization 2505.22323; DeepSeek-V3 aux-free)
- IZ01 Routing-free MoE (experts self-activate, no centralized router) `open`
- IZ02 Path-constrained MoE (concentrated expert paths, 11% lower entropy) `open`
- IZ03 Expert-specialization gradient objective (diversify expert behaviors) `open`
- IZ04 Auxiliary-loss-free balancing (per-expert bias, DeepSeek-V3 style) `open`
- IZ05 Router z-loss (logit regularization for stability) `open`
- IZ06 Token-choice + expert-choice hybrid balancing `open`
- IZ07 Similarity-preserving routers (load balance via expert similarity) `open`
- IZ08 Device-level balancing (per-device expert grouping) `open`
- IZ09 Fine-grained expert dispatch (many small experts, DeepSeek-V2) `open`
- IZ10 Interleaved MoE layers (every 4th/6th layer MoE) `open`
- IZ11 Routing consistency monitor (cross-layer path consistency) `open`
- IZ12 Expert entropy monitor (routing entropy health) `open`
- IZ13 Router perturbation robustness (22.5x robust paths) `open`
- IZ14 Expert-collapse prevention (idle-expert guard) `open`
- IZ15 Load-balance telemetry (per-expert utilization) `open`
- IZ16 Expert specialization score (how distinct are experts) `open`
- IZ17 MoE weight quant (expert weights at low precision, ties quant) `open`
- IZ18 Expert caching (hot-expert weight cache) `open`
- IZ19 MoE speculative decode (draft expert routing) `open`
- IZ20 Expert prefetch (predict next experts, prefetch weights) `open`
- IZ21 MoE energy accounting (per-expert J, ties IJ) `open`
- IZ22 Routing path replay (remember good paths) `open`
- IZ23 Expert load scheduler (batch routing by expert load) `open`
- IZ24 MoE memory tiering (cold experts to slow tier) `open`
- IZ25 Adaptive expert count (grow experts by need) `open`
- IZ26 Router determinism (same input -> same experts) `open`
- IZ27 Expert dropout (train-time expert regularization) `open`
- IZ28 MoE continual learning (new experts for new tasks, ties BB) `open`
- IZ29 Router calibration (router confidence calibration) `open`
- IZ30 Expert routing graph (path visualization) `open`
- IZ31 MoE + Hopfield routing (associative expert selection, ties IP) `open`
- IZ32 Expert weight sharing (shared expert subspaces) `open`
- IZ33 MoE fault tolerance (expert failure fallback) `open`
- IZ34 Router distillation (small router for big MoE) `open`
- IZ35 Expert ensemble (multiple experts for one token) `open`
- IZ36 MoE load-balance benchmark (routing fairness evals) `open`
- IZ37 Expert pruning (drop redundant experts) `open`
- IZ38 MoE token-budget (per-token expert budget) `open`
- IZ39 Routing-aware KV (expert-specific KV partitioning) `open`
- IZ40 Expert importance (which experts matter) `open`
- IZ41 MoE + speculative + paged (combined acceleration) `open`
- IZ42 Router adversarial robustness (routing attacks) `open`
- IZ43 Expert temperature (routing softmax temperature) `open`
- IZ44 MoE incremental experts (add experts online) `open`
- IZ45 Expert-gating MLP (per-expert gating) `open`
- IZ46 MoE telemetry to the operator (routing health) `open`
- IZ47 Expert memory pinning (hot experts in fast memory) `open`
- IZ48 MoE capacity factor tuning (capacity governor) `open`
- IZ49 Router attention (attention-based routing) `open`
- IZ50 Expert credit assignment (which expert helped) `open`
- IZ51 MoE continual specialization (experts specialize over time) `open`
- IZ52 Routing-free activation patterns (AoE/ReMoE comparisons) `open`
- IZ53 Expert load rebalancing (live expert migration) `open`
- IZ54 MoE energy frontier (expert activation vs J) `open`
- IZ55 Router explainability (why this expert) `open`
- IZ56 Expert dedup (merge similar experts) `open`
- IZ57 MoE + agentic routing (task-aware expert selection) `open`
- IZ58 Expert watermark (per-expert provenance) `open`
- IZ59 MoE robustness benchmark (expert perturbation evals) `open`
- IZ60 Router prior (domain-prior routing) `open`
- IZ61 Expert vector cache (expert output cache) `open`
- IZ62 MoE scheduling (expert compute scheduling, ties IR) `open`
- IZ63 Router feedback loop (routing errors -> retrain router) `open`
- IZ64 Expert bias adaptation (bias-based load balancing) `open`
- IZ65 MoE quantization-aware routing (quantized router) `open`
- IZ66 Expert co-activation analysis (which experts fire together) `open`
- IZ67 MoE operator (auto-tune routing config, ties IV) `open`
Status: `open` (67 gaps; routing-free/path-constrained MoE + aux-loss-free balancing + expert specialization)
""")

T.append("""
## Theme JA: Architecture hybrids (attention + SSM)
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (Falcon-H1 hybrid; Hymba hybrid-head; Qwen3-Next GDN+Gated-Attn; Kimi Linear; 2507.12442 SSM characterization)
- JA01 Falcon-H1 parallel hybrid (attention + Mamba2 layers, 256K ctx) `open`
- JA02 Hymba hybrid-head (attention + SSM heads in one layer, 11x KV cut) `open`
- JA03 Qwen3-Next GDN + gated-attention alternation (262K native ctx) `open`
- JA04 SSM-at-scale analysis (57K energy 1492J -> 370J) `open`
- JA05 Hybrid Pareto analysis (accuracy vs TTFT frontier) `open`
- JA06 SSM recall-limitation compensation (attention for precise recall) `open`
- JA07 Hybrid layer-position design (which layers are attention) `open`
- JA08 SSM local + attention global (hybrid receptive fields) `open`
- JA09 Hybrid KV budget (attention layers keep KV, SSM layers don't) `open`
- JA10 Hybrid decode scheduling (per-layer mechanism dispatch) `open`
- JA11 SSM prefill speed (SSM TTFT advantage) `open`
- JA12 Hybrid accuracy-parity evaluation (hybrid >= transformer) `open`
- JA13 SSM energy model at scale (energy vs ctx, ties IJ) `open`
- JA14 Hybrid streaming (SSM constant memory + attention window) `open`
- JA15 Gated-attention long-context stability (hybrid stability) `open`
- JA16 Hybrid reasoning accuracy (long-context reasoning on hybrids) `open`
- JA17 SSM + attention co-training (hybrid training recipe) `open`
- JA18 Hybrid quantization (quantize both mechanisms) `open`
- JA19 SSM state + KV unified cache (one memory system) `open`
- JA20 Hybrid speculative decode (SSM drafter + attention verifier) `open`
- JA21 Hybrid architecture selector (auto-pick hybrid ratio) `open`
- JA22 SSM long-context needle test (hybrid recall evals) `open`
- JA23 Hybrid memory bound (attention window + SSM state) `open`
- JA24 SSM on-device viability (consumer-hardware long context) `open`
- JA25 Hybrid layer ablation (which layers need attention) `open`
- JA26 SSM state size tuning (state dimension) `open`
- JA27 Hybrid context switching (mechanism switch on context) `open`
- JA28 SSM + rotary interaction (position in SSM) `open`
- JA29 Hybrid benchmark harness (attention vs SSM vs hybrid) `open`
- JA30 Hybrid energy Pareto (energy/accuracy curves) `open`
- JA31 SSM numerical stability at scale (state drift) `open`
- JA32 Hybrid prefix caching (prefix in both mechanisms) `open`
- JA33 SSM long-context memory accounting (state bytes) `open`
- JA34 Hybrid token efficiency (mechanism-aware token budget) `open`
- JA35 SSM parallel scan on CPU (efficient scan kernels) `open`
- JA36 Hybrid robustness (perturbation resilience of hybrids) `open`
- JA37 SSM attention-sink equivalents (SSM sink tokens) `open`
- JA38 Hybrid eviction (evict attention KV, keep SSM state) `open`
- JA39 SSM speculative draft (recurrent draft heads) `open`
- JA40 Hybrid alignment (preference-align hybrids, ties IQ) `open`
- JA41 SSM world-model integration (SSM for stateful world, ties IN) `open`
- JA42 Hybrid MoE (MoE layers in hybrid models) `open`
- JA43 SSM multi-modal (SSM for audio/video sequences) `open`
- JA44 Hybrid energy operator (mechanism pick by energy, ties IJ07) `open`
- JA45 SSM state snapshot (checkpoint recurrent state) `open`
- JA46 Hybrid context-length switch (switch mechanism past a length) `open`
- JA47 SSM hardware mapping (scan-friendly layout) `open`
- JA48 Hybrid continual learning (state + weights, ties BB) `open`
- JA49 SSM tokenizer interplay (byte-level state inputs) `open`
- JA50 Hybrid provenance (mechanism attribution per token) `open`
- JA51 SSM quantization at scale (quantized recurrent state) `open`
- JA52 Hybrid verifier (verify across mechanisms) `open`
- JA53 SSM capacity analysis (state capacity vs KV) `open`
- JA54 Hybrid serving (schedule hybrid requests, ties IR) `open`
- JA55 SSM robustness benchmark (hybrid perturbation evals) `open`
- JA56 Hybrid telemetry (per-mechanism counters) `open`
- JA57 SSM memory consolidation (state as memory, ties IP) `open`
- JA58 Hybrid energy ledger (per-mechanism J) `open`
- JA59 SSM long-horizon stability (very-long-context behavior) `open`
- JA60 Hybrid architecture search (auto hybrid design) `open`
- JA61 SSM forgetting (state decay, ties IP05) `open`
- JA62 Hybrid multi-tenant (hybrid cache sharing) `open`
- JA63 SSM differential privacy (state privacy) `open`
- JA64 Hybrid watermark (mechanism-tagged outputs) `open`
- JA65 SSM speculative verification (SSM-verified drafts) `open`
- JA66 Hybrid model portability (run on any hardware) `open`
- JA67 Hybrid operator (auto hybrid config, ties IV) `open`
Status: `open` (67 gaps; attention+SSM hybrids, Hymba hybrid-head, Falcon-H1, energy/accuracy Pareto)
""")

T.append("""
## Theme JB: Multimodal token compression
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (MM token compression survey 2507.20198; VisionSelector; visual-text token efficiency 2026)
- JB01 VisionSelector learnable visual-token selection `open`
- JB02 Visual-text token efficiency (text-as-pixels saves 38-58% decoder tokens) `open`
- JB03 Image token compression (patch merging) `open`
- JB04 Video token compression (temporal redundancy) `open`
- JB05 Audio token compression (spectral redundancy) `open`
- JB06 Cross-modal token alignment (CLIP-style, ties CC03) `open`
- JB07 Visual redundancy detection (similar-patch dedup) `open`
- JB08 Modality-aware KV (per-modality KV budgets) `open`
- JB09 Multimodal attention sparsity (vision tokens sparse attention) `open`
- JB10 Token-compression survey gaps (all surveyed methods) `open`
- JB11 Visual token importance scoring (salience-based retention) `open`
- JB12 Audio-visual fusion compression (joint token compression) `open`
- JB13 Multimodal token budget planner (per-modality budgets) `open`
- JB14 Vision encoder efficiency (ViT patch efficiency) `open`
- JB15 Multimodal eviction (evict low-salience modality tokens, ties IO) `open`
- JB16 Cross-modal prefix (shared multimodal prefix) `open`
- JB17 Visual token streaming (streaming image tokens) `open`
- JB18 Multimodal energy (per-modality J, ties IJ) `open`
- JB19 Visual token dedup (repeated-region suppression) `open`
- JB20 Modality routing (which modality matters per task) `open`
- JB21 Multimodal Hopfield memory (cross-modal patterns, ties IP) `open`
- JB22 Vision-language alignment quality monitor `open`
- JB23 Audio token quantization (compressed audio tokens) `open`
- JB24 Visual token reordering (salience-first ordering) `open`
- JB25 Multimodal compression benchmark (MM token evals) `open`
- JB26 Cross-modal retrieval compression (retrieve + compress) `open`
- JB27 Visual attention sink (vision sink tokens) `open`
- JB28 Multimodal speculative decode (vision draft) `open`
- JB29 Token-efficiency for multimodal (dense modality contexts, ties IY) `open`
- JB30 Visual token provenance (which region produced the token) `open`
- JB31 Multimodal cache sharing (cross-request visual KV reuse) `open`
- JB32 Video frame dedup (temporal frame similarity) `open`
- JB33 Audio-visual token fusion (early fusion compression) `open`
- JB34 Multimodal robustness (adversarial modality input, ties IX) `open`
- JB35 Vision token budget governor (per-image token cap) `open`
- JB36 Multimodal alignment energy (alignment cost) `open`
- JB37 Visual token curriculum (easy->hard visual tasks) `open`
- JB38 Modality-fusion attention (attention over fused modalities) `open`
- JB39 Multimodal memory tiers (modality-tiered memory) `open`
- JB40 Visual compression quality audit (perceptual loss checks) `open`
- JB41 Cross-modal token transfer (text cues -> vision tokens) `open`
- JB42 Multimodal prefix cache (vision prefix reuse) `open`
- JB43 Audio event detection (audio token salience) `open`
- JB44 Multimodal planning (plan over modalities, ties IN) `open`
- JB45 Visual token embedding quant (vision embedding compression) `open`
- JB46 Multimodal continual learning (new modalities, ties BB) `open`
- JB47 Cross-modal adversarial robustness (modality attacks) `open`
- JB48 Multimodal energy operator (modality budget pick) `open`
- JB49 Visual token sampling (salience-based token sampling) `open`
- JB50 Multimodal verifier (cross-modal consistency check) `open`
- JB51 Video temporal compression (frame-rate adaptation) `open`
- JB52 Audio-visual coherence (AV alignment check) `open`
- JB53 Multimodal telemetry (per-modality counters) `open`
- JB54 Visual token watermark (image-region provenance) `open`
- JB55 Cross-modal distillation (vision teacher -> text student) `open`
- JB56 Multimodal alignment drift monitor (ties IM06) `open`
- JB57 Visual context management (visual context budgets) `open`
- JB58 Multimodal speculative verification (cross-modal verify) `open`
- JB59 Token-efficiency-aware vision (fewer tokens, same info) `open`
- JB60 Multimodal OOM safety (modality-budget OOM guard) `open`
- JB61 Visual KV quantization (quantized vision KV, ties IO10) `open`
- JB62 Multimodal scheduler (modality-aware scheduling, ties IR) `open`
- JB63 Cross-modal attention pruning (prune low-cross-attention tokens) `open`
- JB64 Multimodal provenance ledger (auditable modality inputs) `open`
- JB65 Visual token importance model (salience predictor) `open`
- JB66 Multimodal frontier (token-compression quality frontier) `open`
- JB67 Multimodal operator (auto modality config, ties IV) `open`
Status: `open` (67 gaps; vision/audio/video token compression + cross-modal budgets + salience retention)
""")

T.append("""
## Theme JC: Quantization frontier (weights + QAT)
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (1.58-bit QAT bottom-up 2411.05882; 16->1.58 transition ACL-2025; BitNet b1.58; 2-bit 2026)
- JC01 1.58-bit QAT (ternary weights via quantization-aware training) `open`
- JC02 16->1.58 transition schedule (when to switch precision mid-training) `open`
- JC03 BitNet 1.58 regularizer view (ternary as regularization) `open`
- JC04 Weight-only 1.58 inference path `open`
- JC05 Two-phase QAT (full-precision warm-up then quantize) `open`
- JC06 Per-layer precision schedule (layer-adaptive bit width) `open`
- JC07 Activation-aware QAT (quantize with activation ranges) `open`
- JC08 Quantization curriculum (gradually reduce bit width) `open`
- JC09 Ternary GEMV optimization (BitNet-style kernel, ties B03) `open`
- JC10 2-bit QAT (2-bit weights with QAT recovery) `open`
- JC11 QAT gradient handling (straight-through estimators) `open`
- JC12 Quantization-aware KV training (QKV in the loop) `open`
- JC13 Precision transition monitor (when to transition) `open`
- JC14 QAT energy accounting (quantized inference J, ties IJ) `open`
- JC15 Quantized fine-tuning (QAT during fine-tune) `open`
- JC16 Bit-width ablation (per-width accuracy curves) `open`
- JC17 QAT robustness (quantized model robustness) `open`
- JC18 Quantization-aware alignment (align quantized models, ties IQ) `open`
- JC19 Mixed-precision QAT (per-tensor precision) `open`
- JC20 QAT evaluation harness (quantized evals) `open`
- JC21 1.58-bit scaling laws (ternary scaling behavior) `open`
- JC22 QAT convergence speed (quantized training speed) `open`
- JC23 Quantized speculative decode (quantized drafter) `open`
- JC24 QAT + LoRA (quantized LoRA adapters) `open`
- JC25 Quantization-aware distillation (teacher guides quantized student) `open`
- JC26 QAT stability (quantization training stability) `open`
- JC27 Quantized memory footprint (weights + KV + activations) `open`
- JC28 QAT operator (auto bit-width pick, ties IV) `open`
- JC29 Quantized MoE (quantized experts, ties IZ) `open`
- JC30 QAT continual learning (quantized continual learning, ties BB) `open`
- JC31 Ternary attention (quantized attention) `open`
- JC32 QAT precision schedule search (search the schedule) `open`
- JC33 Quantized hybrid models (quantize SSM+attention, ties JA) `open`
- JC34 QAT hardware mapping (quantized kernels per device, ties IS) `open`
- JC35 Quantization-aware embedding (quantized embeddings, ties IT) `open`
- JC36 QAT data selection (which data to quantize-train on) `open`
- JC37 Quantized inference accuracy monitor (perplexity guard) `open`
- JC38 QAT + unlearning (quantized forget, ties IM) `open`
- JC39 Ternary KV (1.58 KV cache, ties IO) `open`
- JC40 QAT reproducibility (seeded quantized training) `open`
- JC41 Quantized long-context (quantized KV for long ctx) `open`
- JC42 QAT energy frontier (bits vs J frontier) `open`
- JC43 Quantization-aware RAG (quantized retrievers) `open`
- JC44 QAT multi-objective (accuracy + energy + size) `open`
- JC45 Quantized world-model (quantized state, ties IN) `open`
- JC46 QAT benchmark suite (quantized training evals) `open`
- JC47 Ternary momentum (quantized optimizer states) `open`
- JC48 Quantized multi-tenant (quantized serving, ties IR) `open`
- JC49 QAT adversarial robustness (quantized model attacks) `open`
- JC50 Quantization-aware speculative (spec with quantized models) `open`
- JC51 QAT transfer (quantize small, transfer to big) `open`
- JC52 Quantized memory consolidation (quantized memory, ties IP) `open`
- JC53 QAT watermark (quantized provenance) `open`
- JC54 Quantized streaming (quantized KV streaming, ties L) `open`
- JC55 QAT precision governor (adaptive precision by loss) `open`
- JC56 Quantized attention kernels (int8 attention) `open`
- JC57 QAT curriculum search (auto curriculum) `open`
- JC58 Quantized tokenizer embeddings (quantized vocab) `open`
- JC59 QAT + speculative + quantized (full-stack quantized) `open`
- JC60 Quantized memory tiers (quantized cold KV, ties A06) `open`
- JC61 QAT energy operator (precision pick by energy) `open`
- JC62 Quantized agentic (quantized agents, ties AD) `open`
- JC63 QAT safety (quantized alignment safety) `open`
- JC64 Quantized telemetry (per-precision counters) `open`
- JC65 QAT fault tolerance (quantized error resilience) `open`
- JC66 Quantized continual self-improvement (quantized RSI, ties IV) `open`
- JC67 QAT frontier (bits/accuracy/energy Pareto) `open`
Status: `open` (67 gaps; 1.58-bit QAT, precision-transition schedules, quantized full-stack integration)
""")

with open("research/INDEX.md", "a") as f:
    f.write("".join(T))
print("part3 appended")
