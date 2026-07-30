# Live Decoder Root Cause Analysis

## Symptom
`gen_text MODEL=/home/wubu/models/Agents-A1-4B "Hello" 8` exits silently (exit 0) with no output.

## Diagnosis
`wubu_model_forward` in `src/wubu_model.c:1306` falls through to the final `else` branch at line 1376:

```c
} else {
    memset(embd, 0, N * model->d_model * sizeof(float));
}
```

This path is taken when **all four embedding sources are NULL**:
1. `model->token_embd`       (F32 in-memory)
2. `model->lazy_embd_raw`    (BF16/F16 mmap'd)
3. `model->use_embedding_file` (external binary)
4. `model->token_embd_q`     (quantized mmap'd)

## Root Cause
For safetensors models (Colonel bridge), the embedding tensor is stored as BF16 in the shard but `wubu_model_safetensors_bridge.c:483` only sets `lazy_embd_raw` when the dtype is **exactly** `ST_DTYPE_BF16` or `ST_DTYPE_F16`. The actual dtype in the shard is `ST_DTYPE_BF16`, but the bridge may be reading it as a different type or the raw pointer may be NULL due to a shard-mapping bug.

## Fix Path
1. Verify the actual dtype in `model-00000-of-00002.safetensors` for `embed_tokens.weight`.
2. Ensure `wubu_shard_raw` returns a valid pointer for BF16 embeddings.
3. Add a fallback path: if `lazy_embd_raw` is NULL but `gguf_ctx` is available, extract embeddings from the GGUF context.

## Verified Working Paths
- `test_real_load` with `MAX_LAYERS=2` succeeds because it uses a minimal fixture.
- `wubu_model_forward_from_embd` works when embeddings are provided externally (prefill path in gen_text).

## Next
Wiring the BF16 embedding path for safetensors bridge is blocked by this issue; not a decode loop bug.