/**
 * compare_ggml_matmul.cpp — SIMPLIFIED
 * Uses known GGUF offsets from gguf_reader output.
 * Compares ggml_mul_mat (quantized) vs SGEMM for expert forward.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <ggml.h>

// Forward declare ggml functions
extern "C" void ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);
extern "C" void dequantize_row_iq2_xxs(const void * x, float * y, int64_t k);
extern "C" void dequantize_row_iq3_xxs(const void * x, float * y, int64_t k);
extern "C" int64_t ggml_time_us(void);

#define D_MODEL 2048
#define D_FF 512
#define BOS 248044

static const char *MODEL_PATH = "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
static const char *EMBD_PATH = "/home/wubu/bytropix/data/qwen36_embeddings_c.bin.raw";

int main() {
    printf("=== ggml Quantized Matmul vs SGEMM ===\n\n");
    
    // Load BOS embedding
    float *x = (float *)malloc(D_MODEL * sizeof(float));
    FILE *ef = fopen(EMBD_PATH, "rb");
    fseek(ef, (int64_t)BOS * D_MODEL * sizeof(float), SEEK_SET);
    fread(x, sizeof(float), D_MODEL, ef);
    fclose(ef);
    
    // Read raw quantized data for expert 0 at known GGUF offsets
    // From gguf_reader: blk.0.ffn_gate_exps.weight data_offset = 767627744
    // total_gate_elems = 2048 * 512 * 256 = 268435456
    // Expert 0 starts at offset 0 within the tensor
    // IQ2_XXS block size: 66 bytes / 256 elements
    // Expert 0 gate raw size: 2048 * 512 / 256 * 66 = 270336 bytes
    int64_t expert_elems = D_MODEL * D_FF;
    int64_t gate_raw_size = expert_elems / 256 * 66;
    int64_t down_raw_size = expert_elems / 256 * 50;  // IQ3_XXS: 50 bytes/256

    // Gate: verified absolute position from gguf_reader
    // gate_exps: abs=767627744, type=IQ2_XXS
    // up_exps:   abs=839660000, type=IQ2_XXS  
    // down_exps: abs=664007136, type=IQ3_XXS
    uint64_t gate_abs_off = 767627744;
    uint64_t up_abs_off   = 839660000;
    uint64_t down_abs_off = 664007136;
    // From the earlier python run: blk.0.ffn_gate_exps and blk.0.ffn_up_exps
    // Both have shape [2048, 512, 256] and type IQ2_XXS
    // Gate is first tensor, up is right after gate in file order
    // From gguf_reader: tensors are in order, so up starts right after gate's data
    
    // Actually let me just read the data from known absolute offsets
    // The raw data blob starts at DATA_BLOB in the file
    // gate_exps data starts at DATA_BLOB + 767627744
    // Let me verify by reading first 16 bytes and comparing with compare_weights tool
    
    FILE *mf = fopen(MODEL_PATH, "rb");
    
    // Read raw data for expert 0 from each tensor
    fseek(mf, gate_abs_off, SEEK_SET);
    uint8_t *gate_raw = (uint8_t *)malloc(gate_raw_size);
    fread(gate_raw, 1, gate_raw_size, mf);
    
    // Up projection (IQ2_XXS, same size as gate)
    fseek(mf, up_abs_off, SEEK_SET);
    uint8_t *up_raw = (uint8_t *)malloc(gate_raw_size);  // Same size as gate (IQ2_XXS)
    fread(up_raw, 1, gate_raw_size, mf);
    
    // Down projection (IQ3_XXS: 50 bytes/block)  
    fseek(mf, down_abs_off, SEEK_SET);
    uint8_t *down_raw = (uint8_t *)malloc(down_raw_size);
    fread(down_raw, 1, down_raw_size, mf);
    
    // For up: it's also IQ2_XXS so same size as gate
    uint8_t raw_first[16];
    fseek(mf, gate_abs_off, SEEK_SET);
    fread(raw_first, 1, 16, mf);
    printf("Expert 0 gate raw[0:16]: ");
    for (int i = 0; i < 16; i++) printf("%02x", raw_first[i]);
    printf("\n");
    fclose(mf);
    printf("Gate raw size: %ld bytes\n", (long)gate_raw_size);
    
    // ====== Step 1: ggml quantized matmul ======
    printf("\n=== ggml Quantized Matmul ===\n");
    
    struct ggml_init_params params = { .mem_size = 256*1024*1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) { fprintf(stderr, "FAIL: ggml_init\n"); return 1; }
    
    // Gate projection: gate_t[D_MODEL, D_FF] quantized IQ2_XXS
    struct ggml_tensor *gate_t = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ2_XXS, D_MODEL, D_FF);
    memcpy(gate_t->data, gate_raw, gate_raw_size);
    
    struct ggml_tensor *inp_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D_MODEL);
    memcpy(inp_t->data, x, D_MODEL * sizeof(float));
    
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    struct ggml_tensor *gate_out = ggml_mul_mat(ctx, gate_t, inp_t);
    ggml_set_output(gate_out);
    ggml_build_forward_expand(gf, gate_out);
    int64_t t1 = ggml_time_us();
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    int64_t t2 = ggml_time_us();
    printf("ggml compute: %ld us\n", (long)(t2-t1));
    
    float *ggml_gate = (float *)gate_out->data;
    printf("ggml gate[0..4]: %.8f %.8f %.8f %.8f %.8f\n",
           ggml_gate[0], ggml_gate[1], ggml_gate[2], ggml_gate[3], ggml_gate[4]);
    int nz = 0;
    for (int i = 0; i < D_FF; i++) if (fabsf(ggml_gate[i]) > 0) nz++;
    printf("ggml gate non-zero: %d / %d\n", nz, D_FF);
    double gn = 0;
    for (int i = 0; i < D_FF; i++) { gn += (double)ggml_gate[i] * ggml_gate[i]; }
    printf("ggml gate std: %.6f  norm: %.6f\n", sqrt(gn/D_FF), sqrt(gn));
    
    // Dequant+SGEMM gate for comparison
    printf("\n--- Dequant+SGEMM gate for comparison ---\n");
    const int n_blocks = D_MODEL / 256;
    float gate_sgemm[D_FF];
    for (int j = 0; j < D_FF; j++) {
        double sum = 0.0;
        for (int k = 0; k < D_MODEL; k++) {
            // Read dequantized value from the raw IQ2_XXS data
            // We can't easily dequant inline, so let's just compare ggml values
            // against what our full model produces
        }
    }
    // For now: print ggml gate values in more detail
    printf("ggml gate non-zero indices (first 10): ");
    int cnt = 0;
    for (int i = 0; i < D_FF && cnt < 10; i++) {
        if (fabsf(ggml_gate[i]) > 0) { printf("%d(%.4f) ", i, ggml_gate[i]); cnt++; }
    }
    printf("\n");
    
    // === SGEMM comparison using ggml's OWN dequant ===
    printf("\n--- SGEMM with ggml's dequantized weights ---\n");
    float *gate_f32 = (float *)malloc(D_MODEL * D_FF * sizeof(float));
    dequantize_row_iq2_xxs(gate_raw, gate_f32, (int64_t)D_MODEL * D_FF);
    
    float gate_sgemm[D_FF];
    for (int j = 0; j < D_FF; j++) {
        double sum = 0.0;
        for (int k = 0; k < D_MODEL; k++)
            sum += (double)gate_f32[k + j * D_MODEL] * (double)x[k];
        gate_sgemm[j] = (float)sum;
    }
    double sn = 0;
    for (int i = 0; i < D_FF; i++) sn += (double)gate_sgemm[i] * gate_sgemm[i];
    printf("SGEMM gate std: %.6f\n", sqrt(sn/D_FF));
    
    // Compare ggml vs SGEMM
    double max_diff = 0, dot=0, na=0, nb=0;
    for (int i = 0; i < D_FF; i++) {
        double d = fabs(ggml_gate[i] - gate_sgemm[i]);
        if (d > max_diff) max_diff = d;
        dot += (double)ggml_gate[i] * gate_sgemm[i];
        na += (double)ggml_gate[i] * ggml_gate[i];
        nb += (double)gate_sgemm[i] * gate_sgemm[i];
    }
    printf("ggml vs SGEMM: max_diff=%.10f cos_sim=%.10f\n", max_diff, dot/(sqrt(na)*sqrt(nb)+1e-30));
    printf("--- %s ---\n", max_diff < 1e-6 ? "IDENTICAL" : 
          (max_diff < 1e-3 ? "CLOSE" : "DIFFERENT"));
    free(gate_f32);
    
    // Up projection
    struct ggml_tensor *up_t = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ2_XXS, D_MODEL, D_FF);
    memcpy(up_t->data, up_raw, gate_raw_size);
    
    struct ggml_tensor *up_out = ggml_mul_mat(ctx, up_t, inp_t);
    ggml_set_output(up_out);
    struct ggml_cgraph *gf2 = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf2, up_out);
    ggml_graph_compute_with_ctx(ctx, gf2, 4);
    
    float *ggml_up = (float *)up_out->data;
    double un = 0;
    for (int i = 0; i < D_FF; i++) un += (double)ggml_up[i] * ggml_up[i];
    printf("ggml up[0..4]: %.8f %.8f %.8f %.8f %.8f  std=%.6f\n",
           ggml_up[0], ggml_up[1], ggml_up[2], ggml_up[3], ggml_up[4], sqrt(un/D_FF));
    
    // Act = silu(gate) * up
    float act[D_FF];
    for (int j = 0; j < D_FF; j++) {
        float g = ggml_gate[j];
        float sg = (g < -80.0f) ? 0.0f : g / (1.0f + expf(-g));
        act[j] = sg * ggml_up[j];
    }
    double an = 0;
    for (int i = 0; i < D_FF; i++) an += (double)act[i] * act[i];
    printf("act std=%.6f\n", sqrt(an/D_FF));
    
    // Down projection: IQ3_XXS, [D_FF, D_MODEL]
    struct ggml_tensor *down_t = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ3_XXS, D_FF, D_MODEL);
    memcpy(down_t->data, down_raw, down_raw_size);
    struct ggml_tensor *act_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D_FF);
    memcpy(act_t->data, act, D_FF * sizeof(float));
    
    struct ggml_tensor *down_out = ggml_mul_mat(ctx, down_t, act_t);
    ggml_set_output(down_out);
    struct ggml_cgraph *gf3 = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf3, down_out);
    ggml_graph_compute_with_ctx(ctx, gf3, 4);
    
    float *output = (float *)down_out->data;
    double out_n = 0;
    for (int i = 0; i < D_MODEL; i++) out_n += (double)output[i] * output[i];
    printf("\nggml expert output[0..4]: %.8f %.8f %.8f %.8f %.8f\n",
           output[0], output[1], output[2], output[3], output[4]);
    printf("ggml output std: %.6f  norm: %.6f\n", sqrt(out_n/D_MODEL), sqrt(out_n));
    
    // Apply 1/sqrt(D_MODEL) scaling
    float inv_sqrt_d = 1.0f / sqrtf((float)D_MODEL);
    printf("After 1/sqrt(D_MODEL)=%.4f scaling: std=%.6f\n",
           inv_sqrt_d, sqrt(out_n/D_MODEL) * inv_sqrt_d);
    
    // Compare with reference: our test_full_moe reported L0 MoE output std=0.0164
    printf("Reference L0 MoE output std: 0.0164\n");
    printf("GGML scaled vs ref ratio: %.4f\n",
           (sqrt(out_n/D_MODEL) * inv_sqrt_d) / 0.0164f);
    
    ggml_free(ctx);
    free(x); free(gate_raw); free(up_raw); free(down_raw);
    return 0;
}
