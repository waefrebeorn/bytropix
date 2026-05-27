# State — May 26, 2026 (MTP OOM Fix + Bootstrap EMA)

## BRANCH: cpu-optimize-may26
**Active fork for CPU optimization work. DO NOT MERGE TO MASTER without user review.**

## Changes This Session

### MTP: OOM Fix (gguf_read_raw_tensor)
- **Root cause:** gen_text_mtp buffered the entire MTP GGUF (11.9GB) on top of main model's 10.9GB buffer = 22.8GB total on 11GB WSL → OOM
- **Fix:** Added `gguf_read_raw_tensor()` to read quantized weight blobs from file individually. `wubu_mtp_load` now copies blk.40 tensors (attn_q, attn_k, attn_v, attn_output, MoE experts) to heap via `gguf_read_raw_tensor` when blob=NULL is passed (no full GGUF buffer).
- **MTP free:** Updated `wubu_mtp_free` to free heap-copied quantized weights when `load_from_blob == false`.

### MTP: Bootstrap EMA Correction
- Added prefill-based bootstrap of the logit correction EMA: runs MTP draft on the last prefill position and initializes `logit_correction[]` with the difference between main and MTP logits.
- Acceptance improved: 8% → 13% → 19% across this session.
- **DA Conclusion:** 19% acceptance is still too low for useful speedup (1.19×). Q8_0 lazy dequant cache would only add ~2-3%, insufficient for 50% target.
- **MTP disabled by default.** Code kept for future DDR5 hardware where overhead is lower.

### New Infrastructure
- `include/gguf_reader.h` + `src/gguf_reader.c`: `gguf_read_raw_tensor()` — read quantized tensor bytes from file or blob, enabling file-backed tensor loading without full GGUF buffer.
- `tools/gen_text_mtp.c`: Fixed model paths, removed `gguf_buffer_data()` call for MTP GGUF.

## Remaining CPU Opportunities (ordered)
1. **Output proj split** — 92ms/token of decode (25%). At DDR4 bandwidth limit, further gains need data reduction or MTP spec decode. Next priority.
2. **MTP** — 19% acceptance, not viable. Revisit with DDR5 or better draft head.
3. **Chunked SSM at CS=1** — Only useful for 256K+ context. No speedup at short lengths.