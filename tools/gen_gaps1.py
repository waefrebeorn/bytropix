#!/usr/bin/env python3
"""KB-7hop sweep part 1: Themes IO (KV eviction), IP (Hopfield memory),
IQ (preference optimization), IR (multi-tenant serving). ~260 gaps."""
T = []

T.append("""
## Theme IO: KV-cache eviction / compression 2026 frontier
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (2603.20397 KV survey; KeyDiff 2504.15364; KVQuant NeurIPS 2024)
- IO01 H2O heavy-hitter token retention (accumulated-attention greedy eviction) `open`
- IO02 StreamingLLM attention-sink keep + rolling window `open` (ties L-theme)
- IO03 SnapKV observation-window pooling + important-prefix retention `open`
- IO04 Proxy-token one-shot eviction (softmax-probability batch discard) `open`
- IO05 InfiniPot novelty distillation (novelty-weighted retain at capacity) `open`
- IO06 HASHEVICT LSH pre-attention eviction (SimHash hamming-distance prune) `open`
- IO07 RocketKV two-stage coarse eviction + dynamic sparse selection `open`
- IO08 Ada-KV head-adaptive budget (eviction-loss upper bound, head-sparse reallocation) `open`
- IO09 KeyDiff key-similarity eviction (attention-sink position varies per head/layer) `open`
- IO10 KVQuant attention-sink-aware quantization + outlier sparse store (3-bit, 4.8x ctx) `open`
- IO11 Semantic-sponsorship KV retention (semantic importance, not score) `open`
- IO12 Pyramidal/block-wise eviction under block prompt processing (eviction-error compounding) `open`
- IO13 Accumulated-attention tracker with per-token running sums (O(1) update) `open`
- IO14 Eviction-loss upper-bound model (formal eviction-error budget) `open`
- IO15 Per-head sink-token discovery (sink position varies across heads/layers) `open`
- IO16 Coarse-to-fine two-stage selection (RocketKV-style page granularity) `open`
- IO17 KV-reconstruction autoencoder importance (regenerate-input criticality) `open`
- IO18 LSH bucket refresh policy (hamming-distance threshold adaptation) `open`
- IO19 Novelty scoring by embedding distance to the retained set `open`
- IO20 Pooled observation window (SnapKV 1D pooling, cluster context) `open`
- IO21 Proxy-token selection via compressed cue (small subset scoring) `open`
- IO22 Eviction + quantization hybrid budget (evict OR compress by value) `open`
- IO23 Sink-token FP16 reservation within quantized caches `open`
- IO24 Outlier channel sparse store (top-1% outlier KV in raw precision) `open`
- IO25 Per-layer eviction budget allocation (attention-sparse vs dispersed layers) `open`
- IO26 Per-head retention count adaptation (variable critical tokens per head) `open`
- IO27 Streaming-aware eviction (evict under continuous generation, not just prefill) `open`
- IO28 Block-boundary eviction coordination (block Xi decisions feed Xi+1) `open`
- IO29 Eviction-error compounding guard (bounded drift across blocks) `open`
- IO30 Key-similarity vs query-similarity dual metric `open`
- IO31 Eviction score normalization across heads (scale-free comparison) `open`
- IO32 Cache-budget renegotiation on OOM (graceful eviction cascade) `open`
- IO33 Hierarchical eviction: hot RAM / warm DRAM / cold NVMe (ties A06) `open`
- IO34 Eviction feedback to the AGI ledger (per-token retention telemetry) `open`
- IO35 Reconstruction-based importance at the page granularity `open`
- IO36 KV-compression ratio governor (target-ratio eviction scheduler) `open`
- IO37 Attention-sink reserve (never evict the first-k tokens regardless of score) `open`
- IO38 LSH distance threshold tuning by observed attention correlation `open`
- IO39 Proxy-token count adaptation by prompt length `open`
- IO40 Eviction-batch grouping (one-shot discard sets, not per-token) `open`
- IO41 Pooling kernel for SnapKV-style context clustering `open`
- IO42 Retention priority queue (heap-based, O(log n) evict) `open`
- IO43 Eviction-aware RoPE (position re-encode after eviction) `open`
- IO44 Compressed-cache correctness audit (perplexity guard after heavy eviction) `open`
- IO45 Eviction decision caching (reuse scores across decode steps) `open`
- IO46 Attention-score streaming aggregator (running softmax without full matrix) `open`
- IO47 Block-paged eviction aligned to the paged-KV table (ties HH02) `open`
- IO48 Importance-vs-novelty dual score (H2O x InfiniPot fusion) `open`
- IO49 Eviction under batched requests (shared cache, per-request criticality) `open`
- IO50 Sink-token count adaptation per model (calibration probe) `open`
- IO51 KVQuant-style 3-bit + outlier split encode/decode kernels `open`
- IO52 Eviction telemetry to the operator (retained-vs-evicted quality delta) `open`
- IO53 Budget-constrained eviction via the energy ledger (ties IJ) `open`
- IO54 Eviction threshold hysteresis (avoid evict/keep oscillation) `open`
- IO55 Cross-session cache reuse (eviction-aware persistence, ties AV03) `open`
- IO56 Semantic eviction via the ANN index (ties AV04) `open`
- IO57 Eviction + speculative-decoding interaction (draft cache retention) `open`
- IO58 Eviction-aware attention scaling (post-eviction normalization) `open`
- IO59 Head-disparity monitor (which heads need the most retention) `open`
- IO60 Eviction policy selector (auto-pick policy by head/block profile) `open`
- IO61 Cache compaction (defragment retained KV pages) `open`
- IO62 Eviction under 1M+ context (cost-modeled retention) `open`
- IO63 Per-layer KV budget governor (layer-wise OOM safety) `open`
- IO64 Eviction-score calibration on a probe set (threshold fitting) `open`
- IO65 Reconstruction-aware eviction in hybrid attention (ties JA) `open`
- IO66 Eviction for multimodal tokens (vision token criticality, ties JB) `open`
- IO67 Eviction ledger integration (which tokens were dropped and why) `open`
Status: `open` (67 gaps; each = a real mechanism from the surveyed literature)
""")

T.append("""
## Theme IP: Hopfield / associative memory 2026 frontier
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (continuous-time Hopfield 2502.10122; dynamic-manifold 2506.01303; federated many-to-one 2603.19902; spectral capacity 2026)
- IP01 Continuous-time memory dynamics (memory state as an ODE, not a discrete update) `open`
- IP02 Dynamic-manifold Hopfield (context-dependent reorganization of the stored manifold) `open`
- IP03 Federated many-to-one Hopfield (heteroassociative: cue -> associated output) `open`
- IP04 Spectral-capacity scaling analysis (capacity vs spectral norm of the memory matrix) `open`
- IP05 Attention-as-Hopfield retrieval formalization (softmax update == memory read) `open`
- IP06 Memory write scheduling (store policy: when a pattern deserves storage) `open`
- IP07 Memory read with beta annealing (sharp-to-flat retrieval over iterations) `open`
- IP08 Pattern separation metric (overlap control between stored patterns) `open`
- IP09 Memory consolidation via rehearsal (periodic re-store of hot patterns) `open`
- IP10 Associative interference monitor (crosstalk detection between similar patterns) `open`
- IP11 Cue denoising with precision control (noisy-cue recall strength) `open`
- IP12 Memory decay scheduler (halflife adaptation by pattern utility) `open`
- IP13 Context-dependent recall gating (context vector modulates the memory read) `open`
- IP14 Heteroassociative binding (input -> output associations, not just auto-assoc) `open`
- IP15 Memory matrix compression (low-rank storage of the pattern matrix) `open`
- IP16 Retrieval by partial cue (prefix / fragment completion) `open`
- IP17 Hopfield-encoded KV cache (attention KV stored as Hopfield patterns, ties IO) `open`
- IP18 Memory capacity accounting (exponential-capacity bookkeeping) `open`
- IP19 Episodic memory with time-tags (temporal associative memory) `open`
- IP20 Memory interference repair (re-orthogonalize similar stored patterns) `open`
- IP21 Continuous-time numerical integration (memory ODE solver, RK4) `open`
- IP22 Manifold curvature estimation for context reorganization `open`
- IP23 Federated memory sharing (patterns shared across agents with provenance) `open`
- IP24 Memory retrieval ranking by spectral overlap `open`
- IP25 Forgetting curve integration (Ebbinghaus curve into the memory weight) `open`
- IP26 Memory replay scheduling (when to replay stored patterns, ties BB) `open`
- IP27 Memory write dedup (identical/duplicate pattern suppression) `open`
- IP28 Memory read with temperature control (softmax sharpness per query) `open`
- IP29 Associative memory for tool selection (cue -> tool pattern retrieval) `open`
- IP30 Memory chaining (sequential pattern association, story recall) `open`
- IP31 Hopfield energy monitor (free-energy of the memory state) `open`
- IP32 Memory stabilization (pattern anchoring after consolidation) `open`
- IP33 Cross-modal associative memory (text cue -> vision pattern, ties JB) `open`
- IP34 Memory corruption detection (pattern degradation watchdog) `open`
- IP35 Memory hygiene: prune low-utility stale patterns (ties IL05) `open`
- IP36 Associative recall in the decode path (memory-guided token candidates) `open`
- IP37 Memory-attention fusion (retrieved pattern as attention bias) `open`
- IP38 Multi-scale memories (short/long-term with separate betas) `open`
- IP39 Memory state snapshot/restore (checkpoint the pattern matrix) `open`
- IP40 Hopfield capacity telemetry (used vs theoretical capacity) `open`
- IP41 Cue embedding quality monitor (cue dims that hurt recall) `open`
- IP42 Memory write batching (bulk store of a session's patterns) `open`
- IP43 Memory read batching (bulk recall for batched decode) `open`
- IP44 Pattern condensation (merge near-identical patterns) `open`
- IP45 Memory-based reasoning (recall chains as CoT memory, ties IV) `open`
- IP46 Associative outlier tolerance (robust recall under adversarial cues) `open`
- IP47 Memory matrix spectral cleanup (drop low-singular-value directions) `open`
- IP48 Context-switch memory isolation (per-task memory partitions) `open`
- IP49 Memory search over patterns (ANN over the memory, ties AV) `open`
- IP50 Memory write/read asymmetry modeling (write cost vs read benefit) `open`
- IP51 Hopfield beta autotuning (temperature fit by recall error) `open`
- IP52 Memory decay vs consolidation arbitration (decay rate vs rehearsal rate) `open`
- IP53 Retrieval-augmented memory (external corpus -> memory patterns) `open`
- IP54 Memory provenance (which source stored each pattern) `open`
- IP55 Memory privacy (forget-set patterns removable, ties IM) `open`
- IP56 Memory load balancing across tiers (hot patterns to fast tier) `open`
- IP57 Associative memory for world-model updates (ties IN) `open`
- IP58 Memory capacity warning (approaching exponential limit) `open`
- IP59 Pattern importance weighting in storage (weighted patterns) `open`
- IP60 Memory coherence across sessions (shared memory merge) `open`
- IP61 Hopfield update with momentum (faster retrieval convergence) `open`
- IP62 Memory read failure handling (no-match fallback policy) `open`
- IP63 Associative memory ablation analysis (which patterns matter) `open`
- IP64 Memory pattern normalization (unit-norm storage for stable recall) `open`
- IP65 Memory-write energy accounting (ties IJ energy ledger) `open`
- IP66 Memory consolidation scheduling (offline consolidation pass) `open`
- IP67 Memory monitor: recall accuracy drift (memory health metric) `open`
Status: `open` (67 gaps; continuous-time / manifold / federated Hopfield + attention-as-memory formalization)
""")

T.append("""
## Theme IQ: Preference optimization frontier
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (2602.00954 small-margin; 2605.20834 DPO/RLHF equivalence; 2509.24159 RE-PO; SimPO; CPO; AlphaPO)
- IQ01 SimPO reference-free reward (length-normalized average log-prob) `open`
- IQ02 CPO conditional preference optimization (difficult discriminative prompts) `open`
- IQ03 IPO identity-preference optimization (squared-error preference loss) `open`
- IQ04 RE-PO robust enhanced policy optimization (general enhancer over DPO/IPO/SimPO/CPO) `open`
- IQ05 AlphaPO reward-shape-aware alignment (reward shaping for DAAs) `open`
- IQ06 Small-margin preference training (margin-aware sampling) `open`
- IQ07 DPO/RLHF conditional-equivalence checker (when DPO == RLHF provably) `open`
- IQ08 Length-bias correction (length-normalized rewards) `open`
- IQ09 Reference-model-free margin (SimPO-style implicit reference) `open`
- IQ10 Preference pair quality weighting (pair difficulty weighting) `open`
- IQ11 Reward accuracy monitor (preference-vs-generation alignment metric) `open`
- IQ12 Preference dataset dedup (near-duplicate pair suppression) `open`
- IQ13 Offline vs online preference mixing (static pairs + live feedback) `open`
- IQ14 Preference aggregation (multiple annotators -> consensus pair) `open`
- IQ15 Margin schedule (margin annealed across training) `open`
- IQ16 Preference noise robustness (label-noise-tolerant loss) `open`
- IQ17 Token-level preference (per-token rewards, not sequence-level) `open`
- IQ18 Step-level process preferences (ties the PRM literature) `open`
- IQ19 Preference cache (reuse pair gradients across updates) `open`
- IQ20 Preference-based early stopping (reward-accuracy gate) `open`
- IQ21 Multi-objective preference (win/lose/ties with three-way loss) `open`
- IQ22 Preference staleness (pair age weighting) `open`
- IQ23 Reward-free calibration (reference-free reward alignment check) `open`
- IQ24 Preference conflict detection (contradictory pairs) `open`
- IQ25 Robust preference optimization (RE-PO-style robustness envelope) `open`
- IQ26 Preference budget allocation (which prompts deserve pairs) `open`
- IQ27 Alignment without forgetting (preference + KL-anchor, ties IM04) `open`
- IQ28 Preference feedback loop to the AGI (user signals as pairs, ties IV) `open`
- IQ29 Implicit reward visualization (reward traces per token) `open`
- IQ30 Preference benchmark harness (alignment eval suite) `open`
- IQ31 Length-normalized margin (SimPO gamma) `open`
- IQ32 Reference-model distillation into the reward (offline reward model) `open`
- IQ33 Preference pair augmentation (synthetic pairs from rejected samples) `open`
- IQ34 Alignment drift monitor during fine-tune (ties IM06) `open`
- IQ35 Preference transfer across domains (pair curriculum) `open`
- IQ36 Reward shaping functions (AlphaPO-style shaping) `open`
- IQ37 Preference update frequency (mini-batch preference mixing) `open`
- IQ38 Pair difficulty-aware sampling (hard-pair emphasis) `open`
- IQ39 Preference-regularized decode (no retrain: preference-constrained sampling) `open`
- IQ40 Alignment energy accounting (preference training under the energy ledger) `open`
- IQ41 Preference-pair provenance (which source made the pair) `open`
- IQ42 Multi-turn preference (conversation-level pairs) `open`
- IQ43 Preference staleness decay (old pairs weight down) `open`
- IQ44 Preference quality gate (reject low-agreement pairs) `open`
- IQ45 DPO vs RLHF divergence metric (when to switch methods) `open`
- IQ46 Preference ensemble (multiple reward hypotheses, ties DD) `open`
- IQ47 Alignment health dashboard (reward accuracy + drift + margin) `open`
- IQ48 Preference-selective replay (alignment replay, ties IM05) `open`
- IQ49 Online preference bootstrap (self-generated pairs, ties IV) `open`
- IQ50 Preference curriculum (easy->hard pair schedule) `open`
- IQ51 Length-robust reward normalization (SimPO's answer) `open`
- IQ52 Preference-aware sampling temperature (confidence-scaled pairs) `open`
- IQ53 Pair margin prediction (predict pair difficulty) `open`
- IQ54 Preference logbook (auditable alignment history) `open`
- IQ55 Alignment verification gate (post-align eval before promotion, ties AX) `open`
- IQ56 Preference transfer learning (align small model, transfer to big) `open`
- IQ57 Reward hacking pre-detection (alignment-time monitoring) `open`
- IQ58 Preference-efficient alignment (fewer pairs via active selection) `open`
- IQ59 Preference entropy (pair distribution flatness) `open`
- IQ60 Alignment + unlearning joint objective (align AND forget, ties IM) `open`
- IQ61 Preference-based model selection (align then pick by eval) `open`
- IQ62 Preference watermark (align-time provenance for outputs) `open`
- IQ63 Preference data versioning (dataset version in the training ledger) `open`
- IQ64 Margin regularization (avoid over-confident preference fitting) `open`
- IQ65 Preference meta-learning (learn the alignment objective, ties IV) `open`
- IQ66 Alignment test-time scaling (preference-guided decoding budget, ties IK) `open`
- IQ67 Preference-to-policy operator (alignment config promotion, ties IM07) `open`
Status: `open` (67 gaps; SimPO/CPO/IPO/RE-PO/AlphaPO + DPO-RLHF equivalence + alignment monitoring)
""")

T.append("""
## Theme IR: Multi-tenant serving / scheduler
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (2603.00356 token management; FIFO-fairness 2026; Stream2LLM MLsys-oral; scheduling survey)
- IR01 Token-management admission control (request acceptance by token budget) `open`
- IR02 Fair-share scheduler (weighted fair queuing over KV budget) `open`
- IR03 Preemption with cache-rebuild cost model (preempt vs restart decision) `open`
- IR04 Activation-budget preemption guard (bounded memory below the threshold) `open`
- IR05 Stream2LLM context streaming + prefill overlap (TTFT reduction) `open`
- IR06 Longest-common-prefix scheduling (minimize redundant prefill) `open`
- IR07 Decoupled scheduling (schedule decision separate from resource acquisition) `open`
- IR08 Hardware-specific cost model for preemption (per-device costs) `open`
- IR09 Burst handling (elastic admission under demand spikes) `open`
- IR10 Priority tiers with starvation bounds `open`
- IR11 Multi-tenant KV isolation (per-tenant cache partitions) `open`
- IR12 Token-budget fairness (each tenant's token share) `open`
- IR13 Preemption victim selection (cheapest-to-restart request) `open`
- IR14 Checkpointed preemption (KV snapshot on preempt, resume not restart) `open`
- IR15 SLO-aware scheduling (per-request latency targets) `open`
- IR16 Batch compaction (fill decode gaps with prefill chunks) `open`
- IR17 Scheduler-cache coherence (schedule decisions respect cache reuse) `open`
- IR18 Dynamic batching window (batch size adaptation by memory) `open`
- IR19 Request-level priority inheritance (ties the OS PI concept) `open`
- IR20 Memory-stability hysteresis (avoid preempt/accept oscillation) `open`
- IR21 Fairness metric monitor (per-tenant service share) `open`
- IR22 Preemption telemetry (preempt frequency, rebuild cost) `open`
- IR23 Co-scheduling prefill+decode (interleaved phases, ties HH04) `open`
- IR24 Cache-aware request routing (route to the node with the prefix) `open`
- IR25 Token-budget profiler (per-request token demand estimation) `open`
- IR26 Admission by predicted KV growth (proactive OOM avoidance) `open`
- IR27 Work-conserving scheduler (never idle while work exists) `open`
- IR28 Preemption budget per tenant (fair preemption) `open`
- IR29 Decode-phase priority (decode > prefill under contention) `open`
- IR30 Scheduler-ledger integration (schedule decisions to the AGI ledger) `open`
- IR31 Multi-queue scheduling (separate queues per SLO class) `open`
- IR32 Backfill scheduling (fill idle slots with background work) `open`
- IR33 Speculative prefill (predict next prompt, prefill ahead) `open`
- IR34 Context-keepalive scheduler (keep hot contexts resident) `open`
- IR35 Eviction-vs-preempt arbitration (evict cold cache or preempt request) `open`
- IR36 Cost-aware scheduling (J/token cost, ties IJ) `open`
- IR37 Scheduler fairness under variable demand (burst-adaptive weights) `open`
- IR38 Request grouping by prefix similarity (batched prefill) `open`
- IR39 Preemption recovery speedup (KV checkpoint restore) `open`
- IR40 Scheduler resilience (scheduler restart without request loss) `open`
- IR41 Multi-tenant security isolation (tenant cache boundaries, ties AD) `open`
- IR42 Token-budget debt tracking (tenant overspend recovery) `open`
- IR43 SLO violation monitor (latency-target breach alerts) `open`
- IR44 Adaptive concurrency (max in-flight by memory pressure) `open`
- IR45 Scheduling policy selector (auto-pick scheduler by load profile) `open`
- IR46 Idle-capacity scavenging (low-priority batch on idle resources) `open`
- IR47 Request coalescing (merge similar prompts) `open`
- IR48 Preemption decision cost-benefit (restart cost vs preempt cost) `open`
- IR49 Memory-pressure feedback loop (scheduler <-> allocator) `open`
- IR50 Deadline-aware scheduling (hard deadlines for time-critical requests) `open`
- IR51 Fair preemption ordering (preempt the least-SLO-critical first) `open`
- IR52 Scheduler benchmarking harness (fairness/latency/throughput evals) `open`
- IR53 Cache-sharing scheduler (shared prefix across tenants with accounting) `open`
- IR54 Preemption-aware token generation (checkpoint generation state) `open`
- IR55 Multi-model scheduling (multiple models on one pool) `open`
- IR56 Scheduler hysteresis (stability under load oscillation) `open`
- IR57 Queue-depth telemetry (per-queue waiting metrics) `open`
- IR58 Request aging (avoid indefinite starvation) `open`
- IR59 Cost-fairness tradeoff scheduler (J/token per tenant, ties IJ) `open`
- IR60 Preemption simulation (dry-run preemption policy) `open`
- IR61 Scheduler config operator (auto-tune scheduler params, ties IV) `open`
- IR62 Token-budget negotiation (tenant request for more budget) `open`
- IR63 Memory-debt reclamation (slow-tenant cache reclaim) `open`
- IR64 Prefill batch planning (chunked prefill schedule) `open`
- IR65 Scheduler event log (auditable schedule decisions) `open`
- IR66 Cross-node scheduling (distributed request placement) `open`
- IR67 Serving energy envelope (power-cap-aware scheduling, ties IJ03) `open`
Status: `open` (67 gaps; fair multi-tenant scheduling + preemption + prefix-aware routing)
""")

with open("research/INDEX.md", "a") as f:
    f.write("".join(T))
print("part1 appended")
