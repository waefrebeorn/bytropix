# Vectors — 7-hop Kevin-Bacon lily-pad KB sweep
## vector search, KV-as-vector, quantization, FlashAttention, similarity, on-device, agentic memory

> Each stone seeds the next hop. Target: map vector-research findings to
> WuBuOS substrate gaps and close the tractable ones as C11 primitives.

## Hop 1 → Vector DB / ANN foundations
Sources: vector indexing strategies (unstructured.io), vector DB architectures
(ACM DL), HNSW vs IVF+PQ comparison, best vector DBs 2026.

Key findings:
- HNSW: high accuracy, fast search, most popular. O(log N) per query.
- IVF: clusters vectors; nprobe controls recall/speed tradeoff.
- PQ: splits vector into M subvectors, each quantized to a codebook entry.
  8–16x compression, ~90-95% recall.
- SQ (Scalar Quantization): fp32→int8/fp16, 75% memory reduction, minimal
  recall loss.
- RaBitQ: 1-bit quantization with correction terms. O(1/sqrt(D)) error bound.
  32x compression, 96%+ recall@10 on DBpedia 768d.
- IVF-PQ: the standard at-scale combo. 64:1 compression, ~70% recall
  (90%+ with lower compression ratio).
- Decision framework: <100K → flat; 100K-1M → HNSW; 1M-10M → IVF-HNSW;
  10M-100M → IVF-PQ; >100M → IVF-PQ + distributed.

WuBuOS gap: `wubu_ctxvm.c` semantic cache is FIFO (no ANN, no quantization).
No HNSW, no PQ, no RaBitQ, no IVF. The KV cache is a flat linear scan.

## Hop 2 → KV cache as vector store
Sources: RAGCache (arXiv 2404.12457), CAG (Cache-Augmented Generation),
persistent KV pipeline, Qdrant semantic cache.

Key findings:
- KV cache IS a vector store: keys are embeddings of token positions; values
  are the attention outputs. Retrieval = similarity search over keys.
- RAGCache stores KV tensors in non-continuous memory blocks for reuse.
  Uses HNSW to search the KV vectors by token similarity.
- CAG preloads essential info into KV cache (no runtime retrieval).
- Persistent KV cache: process context once, persist the processed state;
  repeated queries hit the cache without recomputation.
- Semantic cache: store (query_embedding, response) pairs; on repeated
  queries, retrieve cached response via vector similarity.

WuBuOS gap: `wubu_ctxvm.c` has a FIFO eviction policy, not a vector-similarity
eviction. No KV reuse across sessions. No semantic cache with ANN retrieval.

## Hop 3 → Product quantization / vector compression
Sources: PQ deep-dive (Brenndoerfer, Jan 2026), RaBitQ (Gao & Long, SIGMOD
2024), TurboQuant (Google, ICLR 2026), LanceDB RaBitQ, Milvus PQ/RaBitQ.

Key findings:
- PQ: split D-dim vector into M subvectors of D/M dims each. Each subvector
  quantized to nearest centroid in a codebook. M codebooks of 256 entries.
  Storage: M bytes per vector (vs D*4 bytes). 97% compression at M=16, D=1536.
- RaBitQ: random rotation + binary quantization + correction terms.
  1 bit/dimension. 32x compression. Error O(1/sqrt(D)). Recall 96%+ at 32x.
- SQ8: fp32→int8 per dimension. 75% memory reduction, ~1-2% recall loss.
- Binary quantization: 1 bit/dim. 32x compression. Hamming distance search
  (bitwise XOR + popcount). Fastest query.
- IVF_PQ: cluster + PQ per cluster. Best for billion-scale.
- IVF_RaBitQ: cluster + RaBitQ per cluster. 32x compression, ~95% recall.
- KV cache quantization: FP8 KV halves memory per cached token (vLLM).
  RaBitQ on KV: 32x compression of KV cache for 512K ctx.

WuBuOS gap: no quantization on KV cache. 512K ctx at BF16 = 512K * 2 * 2 * 128
bytes ≈ 256MB per layer. RaBitQ would compress to ~8MB. The FIFO eviction
throws away KV entries without any similarity-based selection.

## Hop 4 → FlashAttention / vectorized attention
Sources: FlashAttention-4 (arXiv 2603.05451), efficient LLM inference,
FlashAttention explained.

Key findings:
- FlashAttention: tiled matmul + online softmax. Never materializes the
  full NxN attention matrix S = QK^T. Instead, processes Q,K,V in blocks
  that fit in SRAM.
- The attention computation is fundamentally a vector operation:
  for each query block, compute similarity to all key blocks, softmax,
  weighted sum of value blocks.
- FlashAttention-4: pipeline matmul and softmax stages, overlap computation
  with HBM reads. Uses shared memory tiling.
- KV cache is a growing vector matrix: each new token appends a K,V row.
  The KV matrix has shape (seq_len, n_heads, head_dim) per layer.
- FP8 KV cache (vLLM): quantize K,V to e4m3 format. Halves memory traffic
  per attention step. Attention computation runs in FP8 with FP32 accumulators.
- At 512K ctx, the KV cache is the binding constraint (not compute).
  FlashAttention's tiling is essential — without it, the NxN attention matrix
  for 512K tokens = 512K^2 * 4 bytes = 1TB (impossible).

WuBuOS gap: `wubu_model.c` KV cache is flat allocation, no tiling, no
quantization. The `stream_kv` is a linear scan. No FlashAttention-style
tiling for 512K ctx.

## Hop 5 → Embedding spaces / similarity metrics
Sources: cosine vs L2 interplay, Matryoshka representation learning,
similarity metrics for vector search, supervised similarity.

Key findings:
- Cosine similarity: angle between vectors. Ignores magnitude. Standard for
  text embeddings.
- L2 (Euclidean): straight-line distance. Accounts for magnitude + direction.
  Standard for image embeddings.
- Dot product: unnormalized cosine. Used when vectors are already normalized.
- Mahalanobis distance: accounts for covariance structure of the embedding
  space. Better than L2 when dimensions are correlated.
- Matryoshka Representation Learning (MRL): a single model produces useful
  embeddings at ANY truncation length. First N dims of a 1024-dim embedding
  form a valid N-dim embedding. Enables flexible-dim retrieval.
- Learned metrics: train a small network to predict relevance scores from
  vector pairs. Better than fixed metrics for domain-specific retrieval.

WuBuOS gap: `wubu_ctxvm.c` semantic cache uses FIFO, not cosine similarity.
No MRL support. No learned metrics. The KV eviction is purely age-based,
not similarity-based (which would keep the most-relevant KV entries).

## Hop 6 → On-device / low-memory vector search
Sources: on-device vector DBs 2026 (ObjectBox), ZVec (SQLite of vector DBs),
Qdrant Edge, best vector DBs 2026 comparison.

Key findings:
- ZVec: "SQLite of vector databases." Embeds directly into application binary,
  runs completely on-device, no server, no network. C library.
- Qdrant Edge: same Qdrant engine, in-process, no API server. Used by robots
  for onboard vector memory (camera → embed → search → decide, all on-device).
- ObjectBox: mobile/embedded vector DB with Java/Swift/Kotlin/C SDKs.
  Supports vector + metadata/hybrid search, incremental CRUD, offline.
- On-device vector search uses IVF-PQ + SQ + binary quantization for low RAM.
  Typical footprint: <50MB for 1M vectors at 768d with RaBitQ.
- Binary quantization + HNSW: fastest on-device ANN. 192 bytes/vector at
  1536d (vs 6KB full float32). 32x compression.
- The "at home" angle: consumer hardware (16-48GB RAM, single GPU) can hold
  10-100M vectors with RaBitQ compression. No server needed.

WuBuOS gap: WuBuOS is an on-device OS (WSL2, consumer hardware). It needs an
on-device vector DB for the KV cache + agentic memory. Currently it has none.

## Hop 7 → Vector memory for agents
Sources: vector databases becoming agentic (LinkedIn), state of vector DBs
Q2 2026 (Actian), agentic memory with vector retrieval.

Key findings:
- Vector DBs are becoming agentic: not just passive retrieval, but active
  memory management for agents. Store experiences as vectors, retrieve
  similar past experiences to inform current decisions.
- Episodic memory as vector index: each experience is embedded; retrieval
  finds the most similar past experience. This is the "remember what worked"
  loop.
- Memory consolidation via vectors: replay important experiences, update
  their embeddings as the agent learns. The vector index IS the agent's
  long-term memory.
- Agentic retrieval: combine vector search with tool use. Retrieve relevant
  past tool outcomes, use them to decide next action.
- The vector memory loop: observe → embed → store → retrieve → decide → act
  → observe. This is the agentic equivalent of the AGI-OS observe→decide→act
  loop, but at the memory level.

WuBuOS gap: `wubu_agentic_mem.c` has episodic→semantic consolidation, but no
vector-based retrieval. Episodic memory is a flat list, not an ANN index.
No agentic retrieval loop (observe→embed→store→retrieve→decide→act).

## Synthesis: WuBuOS vector substrate gaps
1. No ANN index (HNSW/IVF) for KV cache or semantic cache — O(N) scan.
2. No quantization (PQ/RaBitQ/SQ) — KV cache is BF16 full precision.
3. No KV reuse across sessions — FIFO eviction throws away everything.
4. No similarity-based KV eviction — age-based, not relevance-based.
5. No FlashAttention-style tiling — flat attention for 512K ctx.
6. No MRL or flexible-dim embeddings — fixed 128-dim semantic cache.
7. No on-device vector DB — no embedded, offline, low-RAM vector search.
8. No agentic vector memory — episodic memory is a flat list, not an ANN index.

## Action plan: close vector gaps as C11 modules
- wubu_vecsearch.c: HNSW + RaBitQ + on-device vector DB for KV cache +
  agentic episodic memory. Closes gaps 1,2,3,4,5,6,7,8.
- Integrate with wubu_ctxvm.c (semantic cache → ANN-based), wubu_model.c
  (KV cache → RaBitQ-quantized), wubu_agentic_mem.c (episodic → vector index).
- CPU-closable: HNSW is a graph traversal (no GPU needed), RaBitQ is
  integer ops (no GPU needed), on-device vector DB is pure C.
