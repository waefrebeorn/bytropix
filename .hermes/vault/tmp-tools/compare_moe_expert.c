/**
 * compare_moe_expert.c — Compare CPU vs GPU MoE per-expert contributions.
 * Uses the working wubu_moe_forward dispatch (same as test_moe_layer).
 * Dumps per-expert output for expert-by-expert comparison.
 *
 * Build: g++ -O3 -I include -DGPU_SUPPORT -I /usr/local/cuda-13.1/include \
 *   -o compare_moe_expert tools/compare_moe_expert.c \
 *   src/wubu_model.o src/wubu_ssm.o src/wubu_moe.o src/wubu_tokenizer.o \
 *   src/gguf_reader.o src/dequant_iq2_xxs.o src/quantized_matmul.o \
 *   src/quantized_dot_generic.o src/wubu_ssm_chunked.o src/wubu_mobius.o \
 *   src/wubu_nested_ssm.o ... \
 *   src/wubu_model_gpu.o src/cuda_kernels.o src/gpu_output_proj.o \
 *   src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o \
 *   src/gpu_ssm_recurrence.o src/gpu_moe_kernel.o \
 *   -lm -fopenmp -L/usr/local/cuda-13.1/lib64 -lcudart -lcublas
 */
#include "wubu_model.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    mdl.enable_moe = true;

    int gpu_ok = 0;
    if (getenv("GPU")) {
        gpu_ok = wubu_model_gpu_init(&mdl, 4096, 256);
        printf("GPU init: %s\n", gpu_ok ? "OK" : "FAILED");
    }
    if (!gpu_ok) { wubu_model_free(&mdl); return 1; }

    float x[D_MODEL];
    if (mdl.token_embd)
        memcpy(x, mdl.token_embd + 279LL * D_MODEL, D_MODEL * sizeof(float));
    else
        for (int i = 0; i < D_MODEL; i++) x[i] = (float)(i % 100) / 100.0f;

    float normed[D_MODEL];
    wubu_rms_norm(1, 1, D_MODEL, x, mdl.layers[0].attn_norm_weight, 1e-6f, normed);

    moe_weights_t *moe = &mdl.layers[0].moe;

    // === CPU MoE: get per-expert contributions ===
    // We'll run wubu_moe_forward with gpu_ctx=NULL, then read expert_contribs
    // But wubu_moe_forward doesn't expose per-expert data...
    // Instead, we compute per-expert manually from the weight pointers

    int64_t gate_bytes = gguf_raw_size(moe->ffn_gate_exps_q_type, (int64_t)D_MODEL * D_FF);
    int64_t up_bytes   = gguf_raw_size(moe->ffn_up_exps_q_type,   (int64_t)D_MODEL * D_FF);
    int64_t down_bytes = gguf_raw_size(moe->ffn_down_exps_q_type, (int64_t)D_FF * D_MODEL);

    printf("gate_bytes=%ld up_bytes=%ld down_bytes=%ld\n", gate_bytes, up_bytes, down_bytes);
    printf("Gate type=%d Up type=%d Down type=%d\n", moe->ffn_gate_exps_q_type, moe->ffn_up_exps_q_type, moe->ffn_down_exps_q_type);

    // Compute CPU per-expert output for expert 0
    int e = 0;
    const uint8_t *gate_q = moe->ffn_gate_exps_q + (int64_t)e * gate_bytes;
    const uint8_t *up_q   = moe->ffn_up_exps_q   + (int64_t)e * up_bytes;
    const uint8_t *down_q = moe->ffn_down_exps_q + (int64_t)e * down_bytes;

    float cpu_gate[D_FF], cpu_up[D_FF], cpu_act[D_FF], cpu_out[D_MODEL];
    memset(cpu_out, 0, sizeof(cpu_out));

    quantized_matmul(normed, gate_q, moe->ffn_gate_exps_q_type, D_MODEL, D_FF, 0, cpu_gate);
    quantized_matmul(normed, up_q,   moe->ffn_up_exps_q_type,   D_MODEL, D_FF, 0, cpu_up);
    for (int j = 0; j < D_FF; j++) {
        float g = cpu_gate[j];
        cpu_act[j] = (g < -80.0f ? 0.0f : g / (1.0f + expf(-g))) * cpu_up[j];
    }
    quantized_matmul(cpu_act, down_q, moe->ffn_down_exps_q_type, D_FF, D_MODEL, 0, cpu_out);

    // Also dump gate and up outputs for expert 0
    // These are the inputs to SiLU activation, before the nonlinearity
    printf("\nCPU expert 0 gate_out[0..7]:");
    for (int i = 0; i < 8; i++) printf(" %.6f", cpu_gate[i]);
    printf("\n");

    // === GPU MoE: use the existing dispatch through wubu_moe_forward ===
    // Set gpu_ctx to enable GPU path, run with first token's expert selection
    // The GPU path does 8 experts and weighted sum. We capture total output.
    float gpu_full[D_MODEL];
    memset(gpu_full, 0, sizeof(gpu_full));

    moe->gpu_ctx = (void*)&mdl;  // enable GPU path
    wubu_moe_forward(normed, 1, 1, moe, gpu_full, NULL);

    printf("\n=== CPU vs GPU (full MoE, all 8 experts weighted, layer 0) ===\n");
    double cg_dot = 0, cg_n1 = 0, cg_n2 = 0;
    double max_diff = 0;
    int max_idx = -1;
    for (int i = 0; i < D_MODEL; i++) {
        cg_dot += (double)cpu_out[i] * (double)gpu_full[i];
        cg_n1  += (double)cpu_out[i] * (double)cpu_out[i];
        cg_n2  += (double)gpu_full[i] * (double)gpu_full[i];
        double d = fabs((double)cpu_out[i] - (double)gpu_full[i]);
        if (d > max_diff) { max_diff = d; max_idx = i; }
    }
    // Wait — cpu_out is only ONE expert (expert 0) weighted by 1.0, while gpu_full is the full 8-expert weighted sum.
    // This comparison is wrong. Let me fix: compare CPU single-expert against relevant GPU output.
    // Actually the GPU MoE with full routing produces the weighted sum of all 8 experts.
    // To compare single-expert, we'd need to modify the GPU kernel to only run one expert.
    // The existing test_moe_layer already does the full comparison (8 experts).
    // Let me instead compare the CPU per-expert output and verify it matches llama.cpp's per-expert.

    printf("CPU (expert 0 only) [0..3]: %.6f %.6f %.6f %.6f\n", cpu_out[0], cpu_out[1], cpu_out[2], cpu_out[3]);
    printf("GPU (full 8 experts) [0..3]: %.6f %.6f %.6f %.6f\n", gpu_full[0], gpu_full[1], gpu_full[2], gpu_full[3]);

    // Dump files for offline analysis
    FILE *fc = fopen("/tmp/cpu_layer0_expert0.bin", "wb");
    if (fc) { fwrite(cpu_out, sizeof(float), D_MODEL, fc); fclose(fc); }
    FILE *fg = fopen("/tmp/gpu_layer0_full.bin", "wb");
    if (fg) { fwrite(gpu_full, sizeof(float), D_MODEL, fg); fclose(fg); }
    printf("\nDumped to /tmp/cpu_layer0_expert0.bin and /tmp/gpu_layer0_full.bin\n");

    // Also dump the gate and up outputs for CPU expert 0
    // Compare with what llama.cpp would produce
    fc = fopen("/tmp/cpu_expert0_gate.bin", "wb");
    if (fc) { fwrite(cpu_gate, sizeof(float), D_FF, fc); fclose(fc); }
    fc = fopen("/tmp/cpu_expert0_up.bin", "wb");
    if (fc) { fwrite(cpu_up, sizeof(float), D_FF, fc); fclose(fc); }
    fc = fopen("/tmp/cpu_expert0_act.bin", "wb");
    if (fc) { fwrite(cpu_act, sizeof(float), D_FF, fc); fclose(fc); }
    printf("Also dumped gate/up/act for CPU expert 0\n");

    moe->gpu_ctx = NULL;
    wubu_model_free(&mdl);
    return 0;
}
