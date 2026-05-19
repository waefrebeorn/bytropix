# NV64 RDRAM Ring Buffer — CPU/GPU Tandem Sync Architecture

## Concept: Time-Synchronized Ring Buffer

Like N64 RDRAM (Rambus DRAM), hide memory latency by:
1. Accepting a fixed **64-token lag** before first output
2. Prefilling a **ring buffer** with 64 tokens' worth of weight data
3. Pipelining compute: while token N computes, token N+64's weights load
4. CPU + GPU time-synced on **token boundaries** — no sync points mid-token

## Why 64 Tokens

- 2048-dim hidden × 40 layers × 4 bytes = 327KB per token's full activation path
- Model weights: ~11GB (IQ2_M). 11GB / 64 = 172MB per-token prefetch window
- DDR5 bandwidth: ~50GB/s → 172MB in 3.4ms → fits in decode budget
- L3 cache (64MB on Ryzen 7950X): 64 tokens × 1MB/layer = 64MB — just fits
- **Ring buffer size**: 64 slots × per-slot weight footprint

## Architecture

```
                            ┌──────────────┐
        ┌──────────────────►│  Ring Buffer  │◄──────────────────┐
        │                   │  [0..63]      │                   │
        │                   │  Prefetch Wnd │                   │
        │                   └──────┬───────┘                   │
        │                          │                           │
   ┌────┴────┐               ┌─────▼──────┐              ┌────┴────┐
   │ GPU     │◄────sync─────►│ Arbiter    │◄────sync────►│ CPU     │
   │ Compute │   token tick   │ Scheduler  │   token tick │ Prefetch│
   └─────────┘               └────────────┘              └─────────┘
```

### Ring Buffer (RDRAM Channel)

```
Slot layout (one token's worth):
┌─────────────────────────────────────────────────┐
│ Token N: 40 layers × weight pointers + blk.40   │
│   Layer 0: q_weight ptr | k_weight ptr | ...    │
│   Layer 1: q_weight ptr | k_weight ptr | ...    │
│   ...                                           │
│   Token N+63: ...                               │
└─────────────────────────────────────────────────┘
```

- **Circular**: slot[0] wraps to slot[63]. Head = prefetch slot, Tail = compute slot
- **3-zone**: Compute Zone (tail+0..tail+7), Prefetch Zone (head-8..head), Gap (middle)
- **Weight pointers only**: ring stores pointers into GGUF blob, not data copies (11GB too large to duplicate)

### Time-Sync Protocol

Each **token tick** (every ~450ms at 2.2 tok/s on CPU):

```
Phase 0: ACQUIRE (0ms)
  - Arbiter grants token slot N to one consumer
  - CPU or GPU gets exclusive write to output[N]

Phase 1: PREFETCH (overlaps compute)
  - Prefetch agent loads slot head+8 through head+15 into L3/L2
  - _mm_prefetch chain: T0(L1) → T1(L2) → T2(L3) for each weight block
  
Phase 2: COMPUTE (parallel)
  - CPU: Layers 0-19 (SSM-heavy, good on CPU due to OpenMP)
  - GPU: Layers 20-39 (GQA+MoE, good on GPU due to matmul throughput)
  - Both write to shared output[N] via arbiter
  
Phase 3: SYNC (token boundary)
  - Barrier: both CPU and GPU must complete their layers
  - Arbiter advances: tail++, head++ (both modulo 64)
  - Pipeline: CPU starts token N+1 layer 0, GPU starts token N+1 layer 20
```

### GPU Tandem Offload

```
CPU side:                    GPU side:
┌──────────────────┐        ┌──────────────────┐
│ Norm + Attn 0-19 │        │ Norm + Attn 20-39│
│ MoE FFN 0-19     │        │ MoE FFN 20-39    │
│                   │        │                  │
│ → writes h[20]   │        │ → reads h[20]    │
│   to page-locked │        │   from GPU mem   │
│   buffer         │        │                  │
│                   │        │ → output[N] =    │
│                   │        │   final logits   │
└──────────────────┘        └──────────────────┘
       │                           │
       └──────── sync ────────────┘
```

**Key insight**: Split at layer 20 (halfway). CPU writes h[20] to pinned memory. GPU reads h[20] from pinned memory. Zero-copy via CUDA Unified Memory or explicit cudaMemcpyAsync on the sync tick.

**Latency hiding**: GPU starts token N+1's layer 20 while CPU finishes token N's layer 0-19. Overlap = one full token compute time (~450ms).

### NV64 RDRAM "Vroom Vroom" Timing

Like RDRAM's packet-switched memory bus, the ring buffer uses **token-synchronous bursts**:

```
tick 0:  ┌─ACQUIRE──┬─PREFETCH─────────────────┬─COMPUTE───────────────────────┬─SYNC─┐
         │ CPU gets  │ Prefetch slots [0..7]    │ CPU: layers 0-19             │ wait │
         │ slot 0    │ into L3                  │ GPU: layers 20-39            │  GPU │
         └──────────┴──────────────────────────┴───────────────────────────────┴──────┘

tick 1:  ┌─ACQUIRE──┬─PREFETCH─────────────────┬─COMPUTE───────────────────────┬─SYNC─┐
         │ CPU gets  │ Prefetch slots [8..15]   │ CPU: layers 0-19             │ wait │
         │ slot 1    │ into L3                  │ GPU: layers 20-39            │  CPU │
         └──────────┴──────────────────────────┴───────────────────────────────┴──────┘

tick 63: ┌─ACQUIRE──┬─PREFETCH─────────────────┬─COMPUTE───────────────────────┬─SYNC─┐
         │          │ Prefetch slots [0..7]    │                              │      │
         │          │ (wraparound)             │                              │      │
         └──────────┴──────────────────────────┴───────────────────────────────┴──────┘
```

The ring's **64-slot rotation** ensures every weight is in L3 by the time it's computed. The 64-token lag is the time to fill the ring on cold start.

### Distributed Extension

Same ring buffer design, but `slot[i] = machine[i % N]`:

```
Machine 0: Ring slots 0, N, 2N, ...
Machine 1: Ring slots 1, N+1, 2N+1, ...
...
Machine N-1: Ring slots N-1, 2N-1, ...
```

Each machine prefetches its own slots. Arbiter is a distributed consensus (ring token passes around machines). Latency = 64-token lag × (1 + distance_to_next_machine / token_time).

## Implementation Plan

### Phase 9a: Ring Buffer Data Structure
- `#define RING_SIZE 64`
- `ring_slot_t slots[RING_SIZE]` — each slot has weight pointers + output buffer
- `ring_head`, `ring_tail` — atomic ints, mod 64

### Phase 9b: Prefetch Agent
- Background thread: `while(genning) { prefetch_next_8_slots(); }`
- Uses `_mm_prefetch` with graduated hints (T2→T1→T0)
- Prefetches GGUF blob offsets directly (avoids copy)

### Phase 9c: Arbiter/Scheduler
- `acquire_slot()` — blocks until slot is prefetched and previous consumer done
- `release_slot(output_ptr)` — signals slot done, advances tail
- Token tick barrier: both CPU and GPU must release before next acquire

### Phase 9d: GPU Tandem
- CUDA kernel for layers 20-39
- CPU writes h[20] to page-locked memory, GPU reads via cudaMemcpyAsync
- GPU output projection (matmul already on GPU)
- Sync via CUDA events on token boundary

## Expected Performance (CPU+GPU tandem)

| Config | Decode | Speedup |
|--------|--------|---------|
| CPU only (current) | 2.1 tok/s | 1× |
| CPU + GPU tandem (half each) | 4.0 tok/s | 1.9× |
| CPU + GPU tandem + ring prefetch | 5.5 tok/s | 2.6× |
| 2-machine distributed (4× CPU) | 8.0 tok/s | 3.8× |

## Memory Budget

- Ring buffer slots: 64 × 512KB per slot = 32MB (weight pointers only)
- GGUF blob: 11GB (already loaded, shared across all slots)
- Output buffers: 64 × 248320 × 4 bytes = 63MB
- Page-locked for GPU: 2048 × 4 bytes = 8KB per transfer (negligible)
