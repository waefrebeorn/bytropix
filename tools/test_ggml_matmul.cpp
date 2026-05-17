/**
 * test_ggml_matmul.c — Compare our matmul against ggml_mul_mat for one expert.
 * Build: g++ -std=c++11 -O2 -I /home/wubu/llama.cpp/include \
 *        -I /home/wubu/llama.cpp/ggml/include -o test_ggml_matmul \
 *        tools/test_ggml_matmul.cpp \
 *        -L /home/wubu/llama.cpp/build/bin -lllama -lggml-base -lggml-cpu -lggml \
 *        -lm -Wl,-rpath,/home/wubu/llama.cpp/build/bin
 *
 * Loads one expert's dequantized gate weights, applies ggml_mul_mat,
 * compares with our manual matmul.
 */
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// Same as our MoE code
#define D_MODEL 2048
#define D_FF 512

int main() {
    // Load expert 0 gate weights (dumped by our dump_moe_for_python)
    FILE *f = fopen("/tmp/moe_exp0_gate.bin", "rb");
    if (!f) { fprintf(stderr, "Run dump_moe_for_python first\n"); return 1; }
    
    std::vector<float> gate_w(D_MODEL * D_FF);
    fread(gate_w.data(), sizeof(float), D_MODEL * D_FF, f);
    fclose(f);
    
    // Load input
    f = fopen("/tmp/moe_test_input.bin", "rb");
    std::vector<float> x(D_MODEL);
    fread(x.data(), sizeof(float), D_MODEL, f);
    fclose(f);
    
    // Method 1: Our manual matmul
    // gate_j = sum_k x_k * gate_w[k + j * D_MODEL]
    std::vector<float> our_gate(D_FF, 0.0f);
    for (int j = 0; j < D_FF; j++) {
        double sum = 0.0;
        for (int k = 0; k < D_MODEL; k++)
            sum += (double)x[k] * (double)gate_w[k + j * D_MODEL];
        our_gate[j] = (float)sum;
    }
    
    // Method 2: ggml_mul_mat
    // Build weight tensor with ggml's layout
    struct ggml_init_params params = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context *ctx = ggml_init(params);
    
    // Weight: [ne0, ne1] = [input_dim, output_dim] in ggml (transposed storage!)
    // ne0=input_dim=D_MODEL (innermost), ne1=output_dim=D_FF
    // Each expert's weight matrix is [output_dim, input_dim] in standard math
    // but ggml stores it transposed: offset = input_idx + output_idx * input_dim
    // This matches our access pattern: gate_w[k + j * D_MODEL]
    struct ggml_tensor *w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_MODEL, D_FF);
    // Copy data: ggml stores ne[0] innermost = k + j * 2048
    // Our gate_w is in the same layout: gate_w[k + j * D_MODEL]
    // So direct memcpy is correct!
    memcpy(w->data, gate_w.data(), D_MODEL * D_FF * sizeof(float));
    
    // Input: [input_dim=D_MODEL, batch=1] → ne0=D_MODEL (matches w.ne[0]), ne1=1
    struct ggml_tensor *inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_MODEL, 1);
    memcpy(inp->data, x.data(), D_MODEL * sizeof(float));
    
    // Compute: result = ggml_mul_mat(w, inp) = [D_FF, 1]
    struct ggml_tensor *result = ggml_mul_mat(ctx, w, inp);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    
    float *ggml_gate = (float *)result->data;
    
    // Compare
    double max_diff = 0.0;
    int max_idx = -1;
    for (int j = 0; j < D_FF; j++) {
        double diff = fabs(our_gate[j] - ggml_gate[j]);
        if (diff > max_diff) { max_diff = diff; max_idx = j; }
    }
    
    double our_rms = 0, ggml_rms = 0;
    for (int j = 0; j < D_FF; j++) {
        our_rms += our_gate[j] * our_gate[j];
        ggml_rms += ggml_gate[j] * ggml_gate[j];
    }
    our_rms = sqrt(our_rms / D_FF);
    ggml_rms = sqrt(ggml_rms / D_FF);
    
    double dot = 0, n1 = 0, n2 = 0;
    for (int j = 0; j < D_FF; j++) {
        dot += our_gate[j] * ggml_gate[j];
        n1 += our_gate[j] * our_gate[j];
        n2 += ggml_gate[j] * ggml_gate[j];
    }
    double cos_sim = dot / sqrt(n1 * n2);
    
    fprintf(stderr, "Our:    rms=%.6f  first 8=[", our_rms);
    for (int j = 0; j < 8 && j < D_FF; j++) fprintf(stderr, "%.4f ", our_gate[j]);
    fprintf(stderr, "]\n");
    
    fprintf(stderr, "GGML:   rms=%.6f  first 8=[", ggml_rms);
    for (int j = 0; j < 8 && j < D_FF; j++) fprintf(stderr, "%.4f ", ggml_gate[j]);
    fprintf(stderr, "]\n");
    
    fprintf(stderr, "Cos-sim: %.10f\n", cos_sim);
    fprintf(stderr, "Max diff: %.10f at index %d\n", max_diff, max_idx);
    
    ggml_free(ctx);
    return 0;
}
