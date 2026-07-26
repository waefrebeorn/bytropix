# vault/

Quantization format references and archived documentation. Reference data only;
no executable code.

## Quantization reference

Formats implemented in `gguf_reader.h` / `src/quantized_dot_generic.c`:

| Format | Bits/weight | Used for |
|--------|:-----------:|----------|
| Q4_K | 5.0 | Output projection, MoE down (3L) |
| Q5_K | 6.5 | SSM attn_qkv/gate, GQA attn_q/k/v/output, shared gate/up |
| Q6_K | 7.5 | SSM ssm_out, shared down |
| IQ2_XXS | 2.2 | MoE expert gate/up (all 40L) |
| IQ3_XXS | 3.3 | MoE expert down (37L) |
| IQ4_XS | 4.3 | MoE expert down (3L) |
| Q8_0 | 9.0 | Quantized activation for matmul |

## Files

| File | Contents |
|------|----------|
| `unsloth-quantization-format.md` | Unsloth UD quantization format. |
| `api-server.md` | API server sandbox notes. |
| `LEGACY.md` | Legacy documentation index. |
| `cache-compression-resources.md` | KV-cache compression references. |
| `bins/` | Archived session snapshots. |

## Related

- `THEORY/` — research papers and runnable math proofs (hyperbolic, GAAD, DFT/DCT).
- `draftPY/` — Python research prototypes for the encoders above.
