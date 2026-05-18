/**
 * compare_quant_matmul_vs_sgemm.c
 *
 * Load one expert's gate weights (IQ2_XXS), compute the expert forward
 * via TWO paths and compare:
 * 
 * Path A (SGEMM): dequant → F32 → matmul (current approach)
 * Path B (ggml):  ggml_mul_mat on quantized IQ2 data (reference approach)
 *
 * Links against libggml.so for the quantized matmul.
 * Compile: gcc -O3 -march=native -fopenmp -I include \
 *   -I ~/llama.cpp/ggml/include \
 *   -o compare_quant_matmul_vs_sgemm \
 *   tools/compare_quant_matmul_vs_sgemm.c \
 *   src/gguf_reader.o src/dequant_iq2_xxs.o src/dequant_iq3_xxs.o \
 *   -L ~/llama.cpp/build/bin -lggml -lggml-base -lggml-cpu \
 *   -lm -fopenmp -Wl,-rpath,$HOME/llama.cpp/build/bin
 */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ggml.h>

// Forward declare dequant functions
void dequantize_iq2_xxs_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq3_xxs_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_iq4_xs_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_q5_k_row(const uint8_t *data, float *output, int64_t n_elems);
void dequantize_q6_k_row(const uint8_t *data, float *output, int64_t n_elems);

#define D_MODEL 2048
#define D_FF 512
#define N_EXPERTS 256
#define N_ACTIVE_EXPTS 8
#define EXPERT 0  // Test expert 0

static void moe_expert_forward_sgemm(
    const float *x,          // [D_MODEL]
    const float *gate_weight, // [D_MODEL, D_FF]
    const float *up_weight,   // [D_MODEL, D_FF]
    const float *down_weight, // [D_FF, D_MODEL]
    float *output)            // [D_MODEL]
{
    float gate[D_FF], up[D_FF], act[D_FF];
    
    // gate = x @ gate_weight
    for (int j = 0; j < D_FF; j++) {
        double sum = 0.0;
        for (int k = 0; k < D_MODEL; k++)
            sum += (double)x[k] * (double)gate_weight[k + j * D_MODEL];
        gate[j] = (float)sum;
    }
    
    // up = x @ up_weight
    for (int j = 0; j < D_FF; j++) {
        double sum = 0.0;
        for (int k = 0; k < D_MODEL; k++)
            sum += (double)x[k] * (double)up_weight[k + j * D_MODEL];
        up[j] = (float)sum;
    }
    
    // act = silu(gate) * up
    for (int j = 0; j < D_FF; j++) {
        float g = gate[j];
        float silu_g = (g < -80.0f) ? 0.0f : g / (1.0f + expf(-g));
        act[j] = silu_g * up[j];
    }
    
    // output = act @ down_weight
    for (int j = 0; j < D_MODEL; j++) {
        double sum = 0.0;
        for (int k = 0; k < D_FF; k++)
            sum += (double)act[k] * (double)down_weight[k + j * D_FF];
        output[j] = (float)sum;
    }
}

int main() {
    printf("=== Quantized Matmul vs SGEMM Comparison ===\n\n");
    
    // Load model
    gguf_ctx *ctx = gguf_open("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf");
    if (!ctx) return 1;
    
    // ====== Step 1: Load test input ======
    // Use BOS embedding as input (like the model does)
    float *x = (float *)malloc(D_MODEL * sizeof(float));
    FILE *f = fopen("/home/wubu/bytropix/data/qwen36_embeddings_c.bin.raw", "rb");
    if (!f) { printf("FAIL: can't open embeddings\n"); return 1; }
    fseek(f, 248044LL * D_MODEL * sizeof(float), SEEK_SET);
    fread(x, sizeof(float), D_MODEL, f);
    fclose(f);
    
    // ====== Step 2: Load expert weights via gguf_reader (dequant → F32) ======
    gguf_tensor_info *t_gate = gguf_find_tensor(ctx, "blk.0.ffn_gate_exps.weight");
    gguf_tensor_info *t_up   = gguf_find_tensor(ctx, "blk.0.ffn_up_exps.weight");
    gguf_tensor_info *t_down = gguf_find_tensor(ctx, "blk.0.ffn_down_exps.weight");
    
    if (!t_gate || !t_up || !t_down) return 1;
    
    int64_t n_gate = t_gate->dims[0] * t_gate->dims[1] * t_gate->dims[2];
    int64_t n_up   = t_up->dims[0] * t_up->dims[1] * t_up->dims[2];
    int64_t n_down = t_down->dims[0] * t_down->dims[1] * t_down->dims[2];
    
    // Path A: Dequant → F32 weights
    float *gate_f32 = (float *)malloc(n_gate * sizeof(float));
    float *up_f32   = (float *)malloc(n_up * sizeof(float));
    float *down_f32 = (float *)malloc(n_down * sizeof(float));
    
    gguf_read_tensor_f32(ctx, t_gate, gate_f32, n_gate);
    gguf_read_tensor_f32(ctx, t_up,   up_f32,   n_up);
    gguf_read_tensor_f32(ctx, t_down, down_f32, n_down);
    
    // Expert 0 offset
    int64_t e_off_gate = (int64_t)EXPERT * D_MODEL * D_FF;  // 0
    int64_t e_off_down = (int64_t)EXPERT * D_FF * D_MODEL;  // 0
    
    // ====== Step 3: Run SGEMM expert forward ======
    float output_sgemm[D_MODEL];
    moe_expert_forward_sgemm(x,
        gate_f32 + e_off_gate,
        up_f32   + e_off_gate,
        down_f32 + e_off_down,
        output_sgemm);
    
    printf("SGEMM output[0..4]: %.8f %.8f %.8f %.8f %.8f\n",
           output_sgemm[0], output_sgemm[1], output_sgemm[2], output_sgemm[3], output_sgemm[4]);
    double sgemm_norm = 0;
    for (int i = 0; i < D_MODEL; i++) sgemm_norm += (double)output_sgemm[i] * (double)output_sgemm[i];
    printf("SGEMM std: %.6f\n", sqrt(sgemm_norm / D_MODEL));
    
    // ====== Step 4: Load raw IQ2 data ======
    int64_t gate_raw_size = gguf_raw_size(t_gate->type, n_gate);
    int64_t down_raw_size = gguf_raw_size(t_down->type, n_down);
    
    uint8_t *gate_raw = (uint8_t *)malloc(gate_raw_size);
    uint8_t *down_raw = (uint8_t *)malloc(down_raw_size);
    
    // Read raw data from file using gguf offsets
    FILE *mf = fopen("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", "rb");
    if (!mf) return 1;
    
    fseek(mf, ctx->data_blob_offset + t_gate->data_offset, SEEK_SET);
    fread(gate_raw, 1, gate_raw_size, mf);
    
    fseek(mf, ctx->data_blob_offset + t_down->data_offset, SEEK_SET);
    fread(down_raw, 1, down_raw_size, mf);
    fclose(mf);
    
    printf("\nRaw gate tensor: %ld bytes (IQ2_XXS, %ld elems)\n", gate_raw_size, n_gate);
    printf("Raw down tensor: %ld bytes (IQ3_XXS, %ld elems)\n", down_raw_size, n_down);
    
    // ====== Step 5: Dequant and run SGEMM from raw data ======
    float *gate_dequant = (float *)malloc(n_gate * sizeof(float));
    float *down_dequant = (float *)malloc(n_down * sizeof(float));
    
    dequantize_iq2_xxs_row(gate_raw, gate_dequant, n_gate);
    dequantize_iq3_xxs_row(down_raw, down_dequant, n_down);
    
    float output_dequant[D_MODEL];
    moe_expert_forward_sgemm(x,
        gate_dequant + e_off_gate,
        // For up, use same raw path (also IQ2_XXS)
        // For now, use the already-loaded F32 up weights
        up_f32 + e_off_gate,
        down_dequant + e_off_down,
        output_dequant);
    
    // Compare SGEMM (from gguf_reader) vs SGEMM (from raw+dequant)
    double max_diff = 0;
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < D_MODEL; i++) {
        double d = fabs(output_sgemm[i] - output_dequant[i]);
        if (d > max_diff) max_diff = d;
        dot += (double)output_sgemm[i] * (double)output_dequant[i];
        na += (double)output_sgemm[i] * (double)output_sgemm[i];
        nb += (double)output_dequant[i] * (double)output_dequant[i];
    }
    printf("\n=== SGEMM (gguf_reader) vs SGEMM (raw+dequant) ===\n");
    printf("max_diff=%.10f cos_sim=%.10f\n", max_diff, dot/(sqrt(na)*sqrt(nb)+1e-30));
    printf("--- %s ---\n", max_diff < 1e-6 ? "IDENTICAL" : "DIFFERENT");
    
    // ====== Step 6: Use ggml_mul_mat for quantized forward ======
    printf("\n=== Attempting ggml quantized matmul ===\n");
    
    // Initialize ggml
    struct ggml_init_params ggml_params = {
        .mem_size   = 1024 * 1024 * 64,  // 64 MB
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context *ggml_ctx = ggml_init(ggml_params);
    if (!ggml_ctx) { printf("FAIL: ggml_init\n"); return 1; }
    
    // Create tensors for one expert's weights
    // gate_weight: IQ2_XXS, [D_MODEL, D_FF] = [2048, 512]
    int64_t expert_gate_elems = D_MODEL * D_FF;  // 1,048,576
    int64_t expert_gate_bytes = ggml_raw_size(GGML_TYPE_IQ2_XXS, expert_gate_elems);
    
    // Get the raw IQ2 data for expert 0 gate
    int64_t expert_gate_raw_offset = e_off_gate * 66 / 256;  // IQ2_XXS: 66 bytes per 256 elements
    uint8_t *expert_gate_raw_data = gate_raw + expert_gate_raw_offset;
    
    printf("Expert 0 gate: %ld bytes (ggml_raw_size says %ld)\n", 
           (long)(n_gate / 256 * 66), (long)expert_gate_bytes);
    
    // Note: ggml only has column-major quantized tensors, 
    // and creating them from raw data requires the ggml_new_tensor API
    // which allocates memory. We'd need to:
    // 1. Create a ggml tensor of correct type + shape
    // 2. Copy our raw data into it
    // 3. Create an F32 input tensor
    // 4. Call ggml_mul_mat
    // 5. Create a result tensor
    // 6. Build and run the graph
    
    struct ggml_tensor *gate_t = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_IQ2_XXS, D_MODEL, D_FF);
    struct ggml_tensor *up_t   = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_IQ2_XXS, D_MODEL, D_FF);
    struct ggml_tensor *down_t = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_IQ3_XXS, D_FF, D_MODEL);
    
    // Copy raw quantized data into tensors
    memcpy(gate_t->data, expert_gate_raw_data, expert_gate_bytes);
    
    // Input tensor: F32
    struct ggml_tensor *inp_t = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_F32, D_MODEL);
    memcpy(inp_t->data, x, D_MODEL * sizeof(float));
    
    printf("ggml tensors created:\n");
    printf("  gate_t: type=%d ne=[%ld,%ld,1,1]\n", gate_t->type, (long)gate_t->ne[0], (long)gate_t->ne[1]);
    printf("  inp_t:  type=%d ne=[%ld,1,1,1]\n", inp_t->type, (long)inp_t->ne[0]);
    
    // Build computation graph
    struct ggml_cgraph *gf = ggml_new_graph(ggml_ctx);
    
    // gate_out = gate_t @ inp_t  (but ggml_mul_mat computes inp_t @ gate_t with transpose)
    // Actually, ggml_mul_mat(A, B) computes A @ B where both are matrices.
    // A: [M, K], B: [K, N] → result: [M, N]
    // We want: gate_weight[D_MODEL, D_FF]^T @ input[D_MODEL, 1] → gate[1, D_FF]
    // = input[1, D_MODEL] @ gate_weight[D_MODEL, D_FF]
    // With ggml: mul_mat(gate_t, inp_t) where gate_t=[D_MODEL, D_FF], inp_t=[D_MODEL]
    // This computes: gate_inner = inp @ gate_t? 
    // Actually, ggml_mul_mat computes: out = A @ B where A is [D_MODEL, D_FF] and B is [D_MODEL]
    // In GGML convention: https://github.com/ggml-org/ggml/blob/master/docs/ggml.md
    // mul_mat(a, b): result[i,j] = sum_k a[k,i] * b[k,j]
    // So out[j] = sum_k inp_t[k] * gate_t[k, j] 
    // = sum_k x[k] * gate_weight[k, j] ← CORRECT!
    
    struct ggml_tensor *gate_out = ggml_mul_mat(ggml_ctx, gate_t, inp_t);
    
    // Result should be 1D: [D_FF]
    ggml_set_output(gate_out);
    ggml_build_forward_expand(gf, gate_out);
    
    // Compute the graph
    ggml_graph_compute_with_ctx(ggml_ctx, gf, 1);
    
    float *ggml_gate = (float *)gate_out->data;
    printf("\nGGML gate[0..4]: %.8f %.8f %.8f %.8f %.8f\n",
           ggml_gate[0], ggml_gate[1], ggml_gate[2], ggml_gate[3], ggml_gate[4]);
    
    // Compare ggml gate vs SGEMM gate
    // Gate from SGEMM:
    float gate_sgemm[D_FF];
    for (int j = 0; j < D_FF; j++) {
        double sum = 0.0;
        for (int k = 0; k < D_MODEL; k++)
            sum += (double)x[k] * (double)(gate_dequant[e_off_gate + k + j * D_MODEL]);
        gate_sgemm[j] = (float)sum;
    }
    
    dot = 0; na = 0; nb = 0; max_diff = 0;
    for (int i = 0; i < D_FF; i++) {
        double d = fabs(ggml_gate[i] - gate_sgemm[i]);
        if (d > max_diff) max_diff = d;
        dot += (double)ggml_gate[i] * (double)gate_sgemm[i];
        na += (double)ggml_gate[i] * (double)ggml_gate[i];
        nb += (double)gate_sgemm[i] * (double)gate_sgemm[i];
    }
    printf("\n=== GGML gate vs SGEMM gate ===\n");
    printf("max_diff=%.10f cos_sim=%.10f\n", max_diff, dot/(sqrt(na)*sqrt(nb)+1e-30));
    printf("ggml_std=%.6f sgemm_std=%.6f\n", sqrt(na/D_FF), sqrt(nb/D_FF));
    printf("--- %s ---\n", max_diff < 1e-6 ? "IDENTICAL" : 
          (max_diff < 1e-4 ? "CLOSE (fp32 precision)" : "DIFFERENT"));
    
    // Cleanup
    ggml_free(ggml_ctx);
    free(x);
    free(gate_f32); free(up_f32); free(down_f32);
    free(gate_raw); free(down_raw);
    free(gate_dequant); free(down_dequant);
    gguf_close(ctx);
    return 0;
}
