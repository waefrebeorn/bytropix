#!/usr/bin/env python3
"""KB-7hop sweep part 2: Themes IS (PIM/hardware), IT (tokenization),
IU (linear attention), IV (recursive self-improvement), IW (neuromorphic). ~335 gaps."""
T = []

T.append("""
## Theme IS: PIM / near-memory / hardware co-design
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (P3-LLM NPU-PIM 2511.06838; near-memory 3D-DRAM DAC2025; CIM crossbar/RRAM/SRAM 2026; AQPIM HPCA 2026)
- IS01 PIM offload model (which ops move near memory: GEMV over KV) `open`
- IS02 Near-memory KV tier (KV resident next to the compute) `open`
- IS03 Crossbar-compatible matmul emulation (CIM-style GEMV model) `open`
- IS04 SRAM-CIM quantization constraints (bit-cell precision limits) `open`
- IS05 RRAM/FeFET/SOT-MRAM tier model (emerging memory energy/latency) `open`
- IS06 Near-storage compute (smart-SSD KV filter) `open`
- IS07 PIM capacity wall guard (PIM memory budget vs model size) `open`
- IS08 3D-DRAM bonding model (logic-on-memory integration cost) `open`
- IS09 Hybrid NPU-PIM dispatch (when to use PIM vs NPU) `open`
- IS10 Data-movement accounting (bytes moved per op, ties roofline) `open`
- IS11 HBM-stack near-memory buffers (in-stack staging) `open`
- IS12 PIM-friendly weight layout (channel-last for in-memory MAC) `open`
- IS13 Analog-compute noise model (crossbar ADC/DAC precision) `open`
- IS14 Hardware cost model integration (energy+latency per op, ties IJ) `open`
- IS15 PIM offload scheduler (batch ops for near-memory execution) `open`
- IS16 Memory-centric attention tiling (attention tiles resident in memory) `open`
- IS17 Device-model portability (same engine, hardware-abstracted) `open`
- IS18 Near-memory reduce (partial sums at the memory) `open`
- IS19 PIM page-locality (KV pages colocated with the compute) `open`
- IS20 Hardware telemetry model (simulated counters: MACs, bytes, J) `open`
- IS21 CIM bit-precision adaptation (precision per layer by sensitivity) `open`
- IS22 Emerging-memory endurance model (write-wear budget for KV) `open`
- IS23 Near-memory speculative decode (draft heads at the memory) `open`
- IS24 PIM capacity-vs-latency frontier (tradeoff model) `open`
- IS25 Heterogeneous CPU/GPU/NPU-PIM scheduling `open`
- IS26 Memory-wall budget governor (data-movement cap per token) `open`
- IS27 PIM-friendly KV quant (integer KV for CIM, ties IO10) `open`
- IS28 Near-storage RAG (retrieval at the SSD) `open`
- IS29 Hardware abstraction layer for the engine (kernel dispatch table) `open`
- IS30 Crossbar mapping optimizer (weight-to-crossbar placement) `open`
- IS31 PIM energy ledger (in-memory J/op accounting, ties IJ) `open`
- IS32 Near-memory attention sink (sink KV pinned near compute) `open`
- IS33 PIM correctness audit (analog error bounds) `open`
- IS34 Hardware-targeted kernel variants (per-device GEMV) `open`
- IS35 Memory-centric decode loop (decode organized around the memory) `open`
- IS36 PIM-offload benefit predictor (when PIM beats CPU) `open`
- IS37 Near-memory MoE routing (expert weights at the memory) `open`
- IS38 CIM weight stationary layout (weights fixed in crossbar) `open`
- IS39 Hardware counter model (cycle/J/byte counters for tuning) `open`
- IS40 PIM page eviction (KV page movement between tiers) `open`
- IS41 Near-memory prefix cache (LCP prefix at the memory) `open`
- IS42 Emerging-memory latency model (PCM/FeFET read/write costs) `open`
- IS43 PIM-aware batching (batch shapes that fit the memory arrays) `open`
- IS44 Hardware co-simulation harness (simulated device models) `open`
- IS45 PIM numerical stability (low-precision accumulation guards) `open`
- IS46 Near-memory KV compression (compress at the memory, ties IO) `open`
- IS47 Hardware-aware auto-tuning (kernel selection by counters) `open`
- IS48 Memory-centric speculative decode (draft KV near memory) `open`
- IS49 PIM capacity planning (model+KV fit check per device) `open`
- IS50 Near-storage dedup (SSD-side KV dedup) `open`
- IS51 Hardware event simulation (simulated PMU events) `open`
- IS52 PIM dataflow optimization (input-stationary vs output-stationary) `open`
- IS53 Memory-wall roofline update (energy roofline, ties IJ01) `open`
- IS54 Near-memory attention offload (attention compute at DRAM) `open`
- IS55 CIM weight refresh policy (drift compensation) `open`
- IS56 Hardware-aware quantization selector (per-device bit choice) `open`
- IS57 PIM offload regression tests (host parity checks) `open`
- IS58 Near-memory top-k (softmax/selection near memory) `open`
- IS59 Memory-centric scheduling (schedule by memory, ties IR) `open`
- IS60 Hardware diversity matrix (which kernels run where) `open`
- IS61 PIM energy envelope (in-memory power cap, ties IJ03) `open`
- IS62 Near-memory KV dedup (dedup at the memory tier) `open`
- IS63 Hardware cost ledger (J + latency per request) `open`
- IS64 PIM-friendly tokenizer (byte alignment for memory ops) `open`
- IS65 Memory-centric planning (plan steps co-resident with memory) `open`
- IS66 Near-memory verifier (verify tokens near the compute) `open`
- IS67 Hardware-abstracted engine config (device descriptors) `open`
Status: `open` (67 gaps; PIM/CIM/near-memory co-design, hardware-abstracted engine)
""")

T.append("""
## Theme IT: Tokenization / data plane
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (subword decoupling 2604.27263; bit-level BPE 2506.07541; tokenizer-free 2406.19223; lexical density 2026)
- IT01 Bit-level BPE (compression below the byte boundary) `open`
- IT02 Tokenizer-free UTF-8 embeddings (no vocab, ~85% embedding savings) `open`
- IT03 Subword-benefit decoupling (isolate tokenization effects) `open`
- IT04 Byte-entropy-aware merges (low-entropy byte distribution handling) `open`
- IT05 Lexical-density detector (context density -> effective window) `open`
- IT06 Token-merge cache (frequent-token path memoization) `open`
- IT07 Vocabulary pruning (drop unused tokens, remap ids) `open`
- IT08 Tokenizer roundtrip audit (encode/decode fidelity checks) `open`
- IT09 Multi-script tokenization (mixed-script merge policy) `open`
- IT10 Token-level compression (post-token entropy coding) `open`
- IT11 Adaptive tokenization (per-domain vocab) `open`
- IT12 Token-efficiency metric (tokens per information unit) `open`
- IT13 Embedding-table compression (shared embeddings, ties quant) `open`
- IT14 Token-frequency telemetry (vocab usage distribution) `open`
- IT15 Tokenizer-spec versioning (tokenizer changes tracked) `open`
- IT16 OOV handling policy (unknown-token fallbacks) `open`
- IT17 Subword-to-byte fallback (lossless decode guarantees) `open`
- IT18 Token-boundary attention bias (boundary-aware scoring) `open`
- IT19 Token-packing (dense sequence packing for prefill) `open`
- IT20 Byte-level LM adapter (byte model fallback path) `open`
- IT21 Tokenizer benchmark (multilingual token efficiency evals) `open`
- IT22 Token-id remapping (vocab swap without retrain) `open`
- IT23 Token entropy monitor (distribution shift detection) `open`
- IT24 Subword merging heuristics (BPE merge-pair scoring) `open`
- IT25 Tokenizer-cache (memoized encode for repeated text) `open`
- IT26 Unicode-normalization guard (NFKC/NFD handling) `open`
- IT27 Token-length regularization (bounded token growth) `open`
- IT28 Byte-fallback decode (malformed-input recovery) `open`
- IT29 Tokenizer data-flow (token pipeline statistics) `open`
- IT30 Vocabulary merge rules (custom merges for domain terms) `open`
- IT31 Token-pair frequency table (BPE stats) `open`
- IT32 Embedded-token density (lexical density per window, ties IO) `open`
- IT33 Tokenizer determinism (same input -> same ids) `open`
- IT34 Token-budget planner (token estimate before generation, ties IK) `open`
- IT35 Subword-entity alignment (entities spanning tokens) `open`
- IT36 Tokenizer streaming (incremental encode) `open`
- IT37 Vocabulary growth policy (online vocab expansion) `open`
- IT38 Token-space augmentation (token dropout for robustness) `open`
- IT39 Tokenizer energy accounting (encode cost, ties IJ) `open`
- IT40 Byte-level RoPE (position encoding at the byte level) `open`
- IT41 Token-id compression (id entropy coding) `open`
- IT42 Multi-token prediction targets (predict next-N tokens) `open`
- IT43 Token-trie prefix index (fast token prefix lookup) `open`
- IT44 Tokenizer serialization (portable tokenizer format) `open`
- IT45 Tokenization diff tools (compare tokenizer versions) `open`
- IT46 Byte-pair frequency monitor (merge health) `open`
- IT47 Token-efficiency-aware prefill (skip redundant tokens) `open`
- IT48 Tokenizer-free fallback (engine runs without a vocab) `open`
- IT49 Vocabulary coverage metric (OOV rate per domain) `open`
- IT50 Token-boundary watermark (detect token-level tampering) `open`
- IT51 Token sequence compression (lossless token-stream coding) `open`
- IT52 Adaptive byte-vs-subword (per-input path choice) `open`
- IT53 Tokenizer config tuning (merge-threshold autotune) `open`
- IT54 Token metadata (per-token provenance/features) `open`
- IT55 Token embedding quant (embedding-table int8, ties quant) `open`
- IT56 Tokenizer concurrency (thread-safe encode) `open`
- IT57 Token-pair constraints (disallowed merges) `open`
- IT58 Tokenizer fuzz (adversarial byte input, ties IX) `open`
- IT59 Token-efficiency operator (token-budget config pick) `open`
- IT60 Byte-shard alignment (byte-aligned KV pages) `open`
- IT61 Tokenizer profiling (encode/decode timing) `open`
- IT62 Token-stream dedup (repeated-token suppression) `open`
- IT63 Vocabulary pruning safety (never-prune hot tokens) `open`
- IT64 Tokenizer-regression test suite `open`
- IT65 Token-id stability across versions (stable ids) `open`
- IT66 Byte-entropy adaptive merge (entropy-gated merges) `open`
- IT67 Token-efficiency vs quality frontier (compression tradeoff) `open`
Status: `open` (67 gaps; bit-level/byte-level/tokenizer-free tokenization + lexical density)
""")

T.append("""
## Theme IU: Linear attention / fast kernels
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (Mamba3 2603.15569; Kimi Linear/KDA 2510.26692; FLA 2503.14376; Gated DeltaNet; PaTH attention; Hymba hybrid-head)
- IU01 Chunkwise-parallel linear attention (FLA-style chunked formulation) `open`
- IU02 Mamba3 selective state update (recurrent state, constant memory) `open`
- IU03 Gated DeltaNet update (gated delta rule per step) `open`
- IU04 Gated Slot Attention (GSA) state slots `open`
- IU05 HGRN2 gated linear RNN with state expansion `open`
- IU06 GLA hardware-efficient gated linear attention `open`
- IU07 mLSTM sigmoid-gated reduced-compute variant (mLSTMsig) `open`
- IU08 Tiled flash linear attention (TFLA kernel tiling) `open`
- IU09 Lightning attention (Ling-style recurrent linear variant) `open`
- IU10 PaTH position encoding (Householder accumulation) `open`
- IU11 Hybrid-head attention (Hymba-style attention+SSM heads per layer) `open`
- IU12 Hybrid layer mixing (attention/SSM alternation, Falcon-H1 style) `open`
- IU13 SSM KV-cache elimination path (recurrent state instead of KV) `open`
- IU14 SSM long-context scaling (beyond quadratic-attention limits) `open`
- IU15 Hybrid TTFT comparison (SSM 1.35s vs Transformer 8.24s at 57K) `open`
- IU16 Linear-attention numerical stability (recurrent accumulation guards) `open`
- IU17 State compression (learned state summarization) `open`
- IU18 Linear-attention + RoPE interaction (position in linear recurrences) `open`
- IU19 Chunk state transfer (carry chunk states across batches) `open`
- IU20 Gated state decay (forget gates in the state) `open`
- IU21 Delta-rule memory write (delta updates to the state) `open`
- IU22 Linear-attention kernel variant selection (FLA-style autotune) `open`
- IU23 SSM precision control (state precision vs drift) `open`
- IU24 Hybrid energy model (SSM 75% energy cut at 57K, ties IJ) `open`
- IU25 Attention/SSM layer scheduler (which layers are which) `open`
- IU26 Recurrent state checkpoint (state snapshot/restore) `open`
- IU27 Linear-attention recall limits (ICL/precise-recall gap analysis) `open`
- IU28 Hybrid recall compensation (attention layers for precise recall) `open`
- IU29 State-space initialization (SSM parameter init) `open`
- IU30 Linear-attention streaming (constant-memory infinite streaming) `open`
- IU31 Chunked state compute (parallel chunk prefill) `open`
- IU32 Gated linear attention forget schedule (learned gates) `open`
- IU33 Delta-rule binding (write specific keys to state slots) `open`
- IU34 Linear-attention weight tying (recurrent weight sharing) `open`
- IU35 SSM normalization (state normalization for stability) `open`
- IU36 Hybrid decode overlap (attention+SSM heads in one pass) `open`
- IU37 Linear-attention energy ledger (per-state-update J) `open`
- IU38 Recurrent memory decay (state forgetting, ties IP) `open`
- IU39 Linear-attention quantization (quantized state, ties quant) `open`
- IU40 SSM long-context memory bound (constant memory proof) `open`
- IU41 Hybrid benchmark harness (attention vs SSM vs hybrid evals) `open`
- IU42 State expansion ratio tuning (HGRN2-style) `open`
- IU43 Linear-attention speculative decode (recurrent drafter) `open`
- IU44 Chunk parallelization (sequence-chunk parallelism) `open`
- IU45 Gated state multiplexing (shared state across heads) `open`
- IU46 Linear-attention stability monitor (state norm watchdog) `open`
- IU47 Hybrid position encoding (per-head position schemes) `open`
- IU48 SSM hardware mapping (recurrent scan on CPU) `open`
- IU49 Linear-attention gradient path (backward recurrence) `open`
- IU50 Recurrent attention span (effective receptive field) `open`
- IU51 Hybrid layer count tuning (attention/SSM ratio) `open`
- IU52 Linear-attention memory bound (O(1) state size) `open`
- IU53 Delta-rule capacity (state slot capacity, ties IP) `open`
- IU54 SSM multi-scale states (parallel state scales) `open`
- IU55 Linear-attention token-efficiency (ties IT) `open`
- IU56 Hybrid decode scheduling (which phase uses which mechanism) `open`
- IU57 Gated linear attention init (gating init for stability) `open`
- IU58 Linear-attention long-context eval (needle tests) `open`
- IU59 SSM state pruning (drop low-importance state dims) `open`
- IU60 Hybrid attention cost model (attention vs SSM per layer) `open`
- IU61 Linear-attention + Hopfield memory (state as associative memory, ties IP) `open`
- IU62 Chunked linear-attention prefill (parallel chunk prefill) `open`
- IU63 SSM robustness (perturbation sensitivity) `open`
- IU64 Hybrid energy frontier (Pareto energy/accuracy) `open`
- IU65 Linear-attention operator (mechanism selection by context length) `open`
- IU66 Recurrent state ledger (state telemetry) `open`
- IU67 Hybrid model fusion (merge attention + SSM outputs) `open`
Status: `open` (67 gaps; linear attention + hybrid SSM kernels + Hymba-style hybrid heads)
""")

T.append("""
## Theme IV: Recursive self-improvement frontier
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (RSI survey 2607.13104; Goedel agent; LADDER 2503.00735; Promptbreeder; HyperAgents 2603.19461; AUTOHARNESS; ICLR-2026 RSI workshop)
- IV01 Bounded verifiable RSI loops (self-improve with a verifier gate) `open`
- IV02 Goedel-style self-referential agent (improve the improver) `open`
- IV03 LADDER recursive problem decomposition (decompose-and-improve) `open`
- IV04 Promptbreeder prompt evolution (self-referential prompt mutation) `open`
- IV05 HyperAgents metacognitive transfer (improve strategies across domains) `open`
- IV06 AUTOHARNESS code-harness synthesis (auto-generate test harnesses) `open`
- IV07 Intrinsic self-reflection for preference policy (self-reflection in RL) `open`
- IV08 Soft-mellowmax Monte-Carlo planning (softmax-planned search) `open`
- IV09 Experience-learning loop (streaming telemetry -> improvement) `open`
- IV10 Synthetic-data pipeline for self-improvement (self-generated training data) `open`
- IV11 Weak-to-strong generalization loop (small teacher -> big student) `open`
- IV12 Scaffolding improvement (improve the agent framework itself) `open`
- IV13 Full scaffolding search (search the agent design space) `open`
- IV14 Self-awareness audit (the agent knows its own capability) `open`
- IV15 Bounded self-modification (safe-pace weight updates) `open`
- IV16 Continual fine-tuning scheduler (when to schedule fine-tunes) `open`
- IV17 Self-play for improvement (play against yourself, ties GG) `open`
- IV18 Bug-introduction self-training (inject bugs, learn to fix) `open`
- IV19 Production-signal improvement (real usage rewards -> improvement) `open`
- IV20 Reflection-memory (Reflexion-style episodic reflection log) `open`
- IV21 Reflection-diversity guard (avoid local-optima reflections) `open`
- IV22 Self-improvement ledger (auditable improvement history) `open`
- IV23 Improvement-delta metric (did the change help, ties AH13) `open`
- IV24 Recursive decomposition tree (problem -> subproblem tree) `open`
- IV25 Self-evolution verify gate (promote only verified improvements) `open`
- IV26 Metacognitive loop monitor (the improver's own health) `open`
- IV27 Prompt-archive evolution (prompt population + selection) `open`
- IV28 Cross-domain strategy transfer (strategies generalize) `open`
- IV29 Self-reflective data curation (curate your own training data) `open`
- IV30 Improvement rate monitoring (improvement velocity) `open`
- IV31 Self-harness generation (generate your own eval harness) `open`
- IV32 Recursive self-benchmark (benchmark the benchmark) `open`
- IV33 Weak-supervision amplification (weak labels -> strong model) `open`
- IV34 Self-improvement safety envelope (bounded improvement rate) `open`
- IV35 Experience distillation (telemetry -> training examples) `open`
- IV36 Self-modeling (the agent models its own behavior) `open`
- IV37 Improvement credit assignment (which change caused the gain) `open`
- IV38 Self-referential prompt search (prompts that improve prompts) `open`
- IV39 Recursive verification (verify the verifier) `open`
- IV40 Self-improvement cost ledger (improvement J budget, ties IJ) `open`
- IV41 Continual architecture search (self-searching architecture) `open`
- IV42 Self-improvement regression guard (never regress the baseline) `open`
- IV43 Improvement frontier archive (Pareto improvement archive) `open`
- IV44 Self-explanation (the agent explains its own changes) `open`
- IV45 Recursive loop termination (when improvement saturates) `open`
- IV46 Self-improvement telemetry (loop counters to the ledger) `open`
- IV47 Metacognitive calibration (confidence in own improvements) `open`
- IV48 Improvement replay (replay successful improvement steps) `open`
- IV49 Self-distillation (improve by distilling own outputs) `open`
- IV50 Recursive skill acquisition (learn how to learn, ties skills) `open`
- IV51 Self-improvement governance (HITL gates on self-modification) `open`
- IV52 Improvement provenance (which loop produced the change) `open`
- IV53 Self-improvement sandbox (improvements in isolation, ties AX) `open`
- IV54 Recursive prompt optimization (optimize the optimizer's prompts) `open`
- IV55 Self-improvement energy budget (improve under a J cap) `open`
- IV56 Loop convergence detection (improvement plateau detection) `open`
- IV57 Self-improvement portfolio (parallel improvement candidates) `open`
- IV58 Recursive evaluation (evaluate the evaluator) `open`
- IV59 Self-improvement audit trail (append-only improvement log) `open`
- IV60 Improvement rollback (safe revert of a failed improvement) `open`
- IV61 Self-improvement benchmark suite (RSI evaluation harness) `open`
- IV62 Metacognitive transfer monitor (does improvement transfer) `open`
- IV63 Recursive planning (plan the improvement plan) `open`
- IV64 Self-improvement diversity (avoid converging on one trick) `open`
- IV65 Improvement-interaction analysis (which improvements combine) `open`
- IV66 Self-improvement operator (the DA-3 loop as an operator, ties skill) `open`
- IV67 Recursive self-improvement safety audit (the loop's own alignment) `open`
Status: `open` (67 gaps; bounded verifiable RSI loops, Goedel-style self-reference, reflection + metacognition)
""")

T.append("""
## Theme IW: Neuromorphic / SNN cross-over
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (SNN gating ICLR-2026; multi-core neuromorphic train Nature-2026; event-driven 2026)
- IW01 Spike-encoding of tokens (token -> spike train) `open`
- IW02 Event-driven decode (compute only on spikes) `open`
- IW03 SNN energy model (1.05 TFLOPS/W neuromorphic vs GPU) `open`
- IW04 Brain-inspired gating for robustness (SNN gating mechanism) `open`
- IW05 Sparse computation via spike sparsity (55-85% memory-access cut) `open`
- IW06 Multi-core neuromorphic scheduling (parallel spike cores) `open`
- IW07 Membrane-potential accumulator (leaky integrate-and-fire) `open`
- IW08 Spike-based attention (attention on spike events) `open`
- IW09 Neuromorphic KV (KV as synaptic weights) `open`
- IW10 Spike-timing encoding (temporal coding of tokens) `open`
- IW11 SNN-to-ANN conversion (convert trained ANN to SNN) `open`
- IW12 Energy-sparsity correlation (energy saved per sparsity level) `open`
- IW13 Event-driven token selection (spikes gate token processing) `open`
- IW14 Neuromorphic memory (synaptic weight storage) `open`
- IW15 Spike-rate monitoring (activity health) `open`
- IW16 Threshold adaptation (firing threshold tuning) `open`
- IW17 Neuromorphic MoE (expert activation by spikes) `open`
- IW18 Spike-based speculative decode (spike drafter) `open`
- IW19 Neuromorphic energy ledger (J per spike, ties IJ) `open`
- IW20 Spike-train compression (event compression) `open`
- IW21 SNN robustness (noise tolerance of spike codes) `open`
- IW22 Neuromorphic scheduler (event-driven scheduling, ties IR) `open`
- IW23 Spike-based retrieval (associative recall via spikes, ties IP) `open`
- IW24 Membrane decay tuning (leak rate per layer) `open`
- IW25 Neuromorphic weight quant (synaptic weight precision) `open`
- IW26 Event-driven batching (batch on event density) `open`
- IW27 Spike-timing-dependent plasticity (STDP-style memory write) `open`
- IW28 Neuromorphic forward pass (spike forward alternative) `open`
- IW29 Sparse-event attention (attention only on active tokens) `open`
- IW30 SNN training emulation (surrogate gradient) `open`
- IW31 Neuromorphic memory decay (synaptic decay, ties IP05) `open`
- IW32 Spike latency model (event timing overhead) `open`
- IW33 Neuromorphic robustness benchmark (perturbation tests) `open`
- IW34 Event-driven KV eviction (evict on event inactivity) `open`
- IW35 Spike energy accounting (per-spike J model) `open`
- IW36 Neuromorphic prefix cache (spike prefix sharing) `open`
- IW37 Spike-train entropy (information per spike) `open`
- IW38 SNN-to-engine adapter (spike I/O bridge) `open`
- IW39 Neuromorphic world-model (spike-based state, ties IN) `open`
- IW40 Event-driven reasoning (reason on sparse events) `open`
- IW41 Spike threshold schedule (threshold annealing) `open`
- IW42 Neuromorphic top-k (spike-based selection) `open`
- IW43 SNN accuracy-energy frontier (Pareto) `open`
- IW44 Event-driven prefill (sparse prefill) `open`
- IW45 Spike-train watermark (event provenance) `open`
- IW46 Neuromorphic cache coherence (spike cache consistency) `open`
- IW47 Spike-based continual learning (online spike learning, ties BB) `open`
- IW48 Neuromorphic attention sink (sink as tonic spiking) `open`
- IW49 Event-driven telemetry (spike counters) `open`
- IW50 SNN mixed-precision (spike + analog hybrid) `open`
- IW51 Neuromorphic energy envelope (power cap on spikes, ties IJ03) `open`
- IW52 Spike-train dedup (redundant event suppression) `open`
- IW53 Neuromorphic KV quant (synaptic KV compression) `open`
- IW54 Event-driven sampling (spike-gated decoding) `open`
- IW55 SNN stability analysis (spike dynamics) `open`
- IW56 Neuromorphic memory consolidation (synaptic replay, ties BB) `open`
- IW57 Spike-based verification (verify on spikes) `open`
- IW58 Neuromorphic model selector (SNN vs ANN by task) `open`
- IW59 Event-driven context management (spike context budgets) `open`
- IW60 Spike-train augmentation (event dropout) `open`
- IW61 Neuromorphic error handling (spike fault tolerance) `open`
- IW62 Event-driven RL (spike rewards, ties GG) `open`
- IW63 SNN benchmark harness (energy/accuracy evals) `open`
- IW64 Neuromorphic provenance (spike-source tracking) `open`
- IW65 Event-driven energy operator (spike budget pick, ties IJ07) `open`
- IW66 Spike-based alignment (preference on spikes, ties IM) `open`
- IW67 Neuromorphic AGI substrate (event-driven cognitive architecture) `open`
Status: `open` (67 gaps; spike/event-driven crossover, neuromorphic energy, STDP memory)
""")

with open("research/INDEX.md", "a") as f:
    f.write("".join(T))
print("part2 appended")
