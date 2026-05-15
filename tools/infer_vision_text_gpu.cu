/**
 * infer_vision_text_gpu.cu — GPU-accelerated vision→text pipeline
 *
 * Vision encoder on GPU (cuBLAS), text model forward on CPU.
 * Steps: patch_embed(CPU) → pos_embd(CPU) → upload →
 * 27 ViT layers(GPU) → download → spatial merge(CPU) →
 * mm0→GELU→mm2(CPU) → text model forward(CPU)
 */
#include "wubu_model.h"
#include "wubu_vision.h"
#include "cuda_vision.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *vision_path = argc > 2 ? argv[2] : "/mnt/wslg/distro/models/qwen3.6-35b-mmproj-F16.gguf";
    int H = argc > 3 ? atoi(argv[3]) : 256;
    int W = argc > 4 ? atoi(argv[4]) : 256;
    int B = argc > 5 ? atoi(argv[5]) : 1;

    printf("=== GPU Vision→Text Pipeline ===\n"); fflush(stdout);
    double t0 = now_sec();

    // 1. Load model FIRST (before CUDA init to avoid WSL race)
    printf("Loading model...\n"); fflush(stdout);
    wubu_model_t model;
    if (!wubu_model_init(&model, model_path)) return 1;
    printf("Model loaded: %.2fs\n", now_sec() - t0); fflush(stdout);

    // 2. Load vision encoder
    printf("Loading vision encoder...\n"); fflush(stdout);
    vision_encoder_t enc;
    if (!vision_encoder_init(&enc, vision_path)) return 1;
    printf("Vision loaded: %.2fs\n", now_sec() - t0); fflush(stdout);

    // 2. Determine patch sizes
    int C = 3;
    int patch_h = H / V_PATCH_SIZE;
    int patch_w = W / V_PATCH_SIZE;
    int merged_h = patch_h / 2;
    int merged_w = patch_w / 2;
    int n_patches_total = patch_h * patch_w * V_TEMP_PATCH;
    int n_merged = merged_h * merged_w * V_TEMP_PATCH;
    if (n_patches_total > V_MAX_POS) n_patches_total = V_MAX_POS;
    if (n_merged > V_MAX_POS) n_merged = V_MAX_POS;

    printf("  Patches: %d (%dx%d), merged: %d\n", n_patches_total, patch_h, patch_w, n_merged);

    // 3. CUDA init
    cublasHandle_t cublas_h;
    cudaStream_t stream;
    if (cublasCreate(&cublas_h) != CUBLAS_STATUS_SUCCESS) return 1;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    cublasSetStream(cublas_h, stream);

    // 4. Upload ViT layer weights to GPU
    double t1 = now_sec();
    gpu_vision_weights_t gpu_layers[V_N_LAYERS];
    memset(gpu_layers, 0, sizeof(gpu_layers));
    for (int l = 0; l < V_N_LAYERS; l++) {
        if (!enc.layers[l].loaded) continue;
        if (!gpu_vision_upload_layer(&enc.layers[l], &gpu_layers[l])) return 1;
    }
    // LN weights on GPU
    float *d_ln1_w[V_N_LAYERS], *d_ln1_b[V_N_LAYERS], *d_ln2_w[V_N_LAYERS], *d_ln2_b[V_N_LAYERS];
    float *d_post_ln_w, *d_post_ln_b;
    for (int l = 0; l < V_N_LAYERS; l++) {
        cudaMalloc(&d_ln1_w[l], V_HIDDEN * sizeof(float));
        cudaMemcpy(d_ln1_w[l], enc.layers[l].ln1_weight, V_HIDDEN * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_ln1_b[l], V_HIDDEN * sizeof(float));
        cudaMemcpy(d_ln1_b[l], enc.layers[l].ln1_bias, V_HIDDEN * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_ln2_w[l], V_HIDDEN * sizeof(float));
        cudaMemcpy(d_ln2_w[l], enc.layers[l].ln2_weight, V_HIDDEN * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_ln2_b[l], V_HIDDEN * sizeof(float));
        cudaMemcpy(d_ln2_b[l], enc.layers[l].ln2_bias, V_HIDDEN * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&d_post_ln_w, V_HIDDEN * sizeof(float));
    cudaMemcpy(d_post_ln_w, enc.post_ln_weight, V_HIDDEN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_post_ln_b, V_HIDDEN * sizeof(float));
    cudaMemcpy(d_post_ln_b, enc.post_ln_bias, V_HIDDEN * sizeof(float), cudaMemcpyHostToDevice);
    cudaStreamSynchronize(stream);
    printf("GPU weights uploaded: %.2fs\n", now_sec() - t1);

    // 5. Create synthetic image (or load from file)
    float *pixels = (float *)malloc(B * C * H * W * sizeof(float));
    for (int b = 0; b < B; b++)
        for (int c = 0; c < C; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    pixels[(b*C+c)*H*W + y*W + x] = ((x/16 + y/16) % 2) * 0.8f + 0.1f;

    // 6. === CPU: Patch embedding + position embeddings ===
    double t_patch = now_sec();
    float *h_hidden = (float *)calloc(B * n_patches_total * V_HIDDEN, sizeof(float));
    for (int b = 0; b < B; b++) {
        for (int tp = 0; tp < V_TEMP_PATCH; tp++) {
            const float *kernel = (tp == 0) ? enc.patch_embd_weight : enc.patch_embd_weight2;
            for (int ph = 0; ph < patch_h; ph++) {
                for (int pw = 0; pw < patch_w; pw++) {
                    int idx = (b * V_TEMP_PATCH + tp) * (patch_h * patch_w) + ph * patch_w + pw;
                    if (idx >= n_patches_total) break;
                    float *out = h_hidden + idx * V_HIDDEN;
                    memcpy(out, enc.patch_embd_bias, V_HIDDEN * sizeof(float));
                    for (int ky = 0; ky < V_PATCH_SIZE; ky++)
                        for (int kx = 0; kx < V_PATCH_SIZE; kx++)
                            for (int c = 0; c < C; c++) {
                                float pixel = pixels[(b*C+c)*H*W + (ph*16+ky)*W + (pw*16+kx)];
                                const float *k = kernel + (ky*16+kx)*(C*V_HIDDEN) + c*V_HIDDEN;
                                for (int f = 0; f < V_HIDDEN; f++)
                                    out[f] += pixel * k[f];
                            }
                }
            }
        }
        // Position embeddings
        for (int i = 0; i < n_patches_total; i++)
            for (int f = 0; f < V_HIDDEN; f++)
                h_hidden[i * V_HIDDEN + f] += enc.pos_embd_weight[f * V_MAX_POS + i];
    }
    double t_patch_end = now_sec();
    printf("Patch+pos: %.3fs\n", t_patch_end - t_patch);

    // 7. === GPU: 27 ViT layers ===
    float *d_x, *d_out, *d_scratch;
    cudaMalloc(&d_x, B * n_patches_total * V_HIDDEN * sizeof(float));
    cudaMalloc(&d_out, B * n_patches_total * V_HIDDEN * sizeof(float));
    int scratch_sz = n_patches_total * (V_HIDDEN * 3 + V_HIDDEN * 2) * sizeof(float);
    cudaMalloc(&d_scratch, scratch_sz);

    cudaMemcpy(d_x, h_hidden, B * n_patches_total * V_HIDDEN * sizeof(float), cudaMemcpyHostToDevice);
    float *d_in = d_x, *d_tmp = d_out;

    int threads_per_block = 256;
    int blocks = (n_patches_total + threads_per_block - 1) / threads_per_block;

    for (int l = 0; l < V_N_LAYERS; l++) {
        if (!enc.layers[l].loaded) {
            cudaMemcpy(d_tmp, d_in, B * n_patches_total * V_HIDDEN * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            // LayerNorm 1
            layernorm_kernel<<<blocks, threads_per_block, 0, stream>>>(d_in, d_ln1_w[l], d_ln1_b[l],
                                                                        B * n_patches_total, V_HIDDEN, 1e-6f);
            // Attention + FFN
            gpu_vision_layer_forward(cublas_h, stream, &gpu_layers[l],
                                     d_in, B * n_patches_total, d_tmp, d_scratch);
            // LayerNorm 2
            layernorm_kernel<<<blocks, threads_per_block, 0, stream>>>(d_tmp, d_ln2_w[l], d_ln2_b[l],
                                                                        B * n_patches_total, V_HIDDEN, 1e-6f);
        }
        float *swap = d_in; d_in = d_tmp; d_tmp = swap;
    }

    // Post LN
    layernorm_kernel<<<blocks, threads_per_block, 0, stream>>>(d_in, d_post_ln_w, d_post_ln_b,
                                                                B * n_patches_total, V_HIDDEN, 1e-6f);
    cudaStreamSynchronize(stream);
    double t_gpu = now_sec();
    printf("GPU ViT: %.3fs\n", t_gpu - t_patch_end);

    // 8. Download from GPU
    float *h_vit_out = (float *)malloc(B * n_patches_total * V_HIDDEN * sizeof(float));
    cudaMemcpy(h_vit_out, d_in, B * n_patches_total * V_HIDDEN * sizeof(float), cudaMemcpyDeviceToHost);

    // 9. === CPU: Spatial merge (concatenate 2×2) + MMProj ===
    const int V_MERGE_DIM = V_HIDDEN * 4;
    float *vit_embd = (float *)malloc(B * n_merged * V_OUT_HIDDEN * sizeof(float));

    for (int b = 0; b < B; b++) {
        const float *batch_vit = h_vit_out + b * n_patches_total * V_HIDDEN;
        float *batch_merged = (float *)malloc(n_merged * V_MERGE_DIM * sizeof(float));
        memset(batch_merged, 0, n_merged * V_MERGE_DIM * sizeof(float));

        // Spatial merge: concatenate 2×2 neighbors
        for (int tp = 0; tp < V_TEMP_PATCH; tp++) {
            int tb = tp * (patch_h * patch_w);
            for (int mh = 0; mh < merged_h; mh++) {
                for (int mw = 0; mw < merged_w; mw++) {
                    int dst = tp * (merged_h * merged_w) + mh * merged_w + mw;
                    float *dst_row = batch_merged + dst * V_MERGE_DIM;
                    for (int dy = 0; dy < 2; dy++) {
                        for (int dx = 0; dx < 2; dx++) {
                            int src = tb + (mh*2+dy) * patch_w + (mw*2+dx);
                            int co = (dy*2+dx) * V_HIDDEN;
                            const float *src_row = batch_vit + src * V_HIDDEN;
                            for (int f = 0; f < V_HIDDEN; f++)
                                dst_row[co + f] = src_row[f];
                        }
                    }
                }
            }
        }

        // MMProj: mm0 → GELU → mm2 per merged token
        float *batch_out = vit_embd + b * n_merged * V_OUT_HIDDEN;
        if (enc.mm0_weight) {
            for (int i = 0; i < n_merged; i++) {
                const float *row = batch_merged + i * V_MERGE_DIM;
                float mm0_buf[4608];
                for (int j = 0; j < 4608; j++) {
                    double sum = enc.mm0_bias[j];
                    for (int k = 0; k < 4608; k++)
                        sum += (double)row[k] * (double)enc.mm0_weight[k * 4608 + j];
                    mm0_buf[j] = gelu_tanh((float)sum);
                }
                float *out_row = batch_out + i * V_OUT_HIDDEN;
                for (int j = 0; j < V_OUT_HIDDEN; j++) {
                    double sum = enc.mm2_bias[j];
                    for (int k = 0; k < 4608; k++)
                        sum += (double)mm0_buf[k] * (double)enc.mm2_weight[k * V_OUT_HIDDEN + j];
                    out_row[j] = (float)sum;
                }
            }
        } else {
            for (int i = 0; i < n_merged; i++)
                memcpy(batch_out + i * V_OUT_HIDDEN, batch_merged + i * V_MERGE_DIM,
                       (V_MERGE_DIM < V_OUT_HIDDEN ? V_MERGE_DIM : V_OUT_HIDDEN) * sizeof(float));
        }
        free(batch_merged);
    }
    double t_merge = now_sec();
    printf("Merge+MMProj: %.3fs\n", t_merge - t_gpu);

    // 10. Check vision output
    double t_vision = now_sec();
    float min_v=1e30, max_v=-1e30; int nan_c=0;
    int vsize = B * n_merged * V_OUT_HIDDEN;
    for (int i = 0; i < vsize; i++) {
        if (vit_embd[i] < min_v) min_v = vit_embd[i];
        if (vit_embd[i] > max_v) max_v = vit_embd[i];
        if (isnan(vit_embd[i])) nan_c++;
    }
    printf("\nVision output: [%.4f, %.4f] NaN=%d tokens=%d\n", min_v, max_v, nan_c, n_merged);
    printf("  [0:4]: %.4f %.4f %.4f %.4f\n", vit_embd[0], vit_embd[1], vit_embd[2], vit_embd[3]);

    // 11. Run text forward on loaded model
    printf("\n--- Text Model Forward ---\n"); fflush(stdout);
    float *logits = (float *)malloc(B * n_merged * model.vocab_size * sizeof(float));

    double tt0 = now_sec();
    wubu_model_forward_from_embd(&model, vit_embd, B, n_merged, logits);
    double t_text = now_sec() - tt0;

    float min_l=1e30, max_l=-1e30; int nan_l=0;
    int lsize = B * n_merged * model.vocab_size;
    for (int i = 0; i < lsize; i++) {
        if (logits[i] < min_l) min_l = logits[i];
        if (logits[i] > max_l) max_l = logits[i];
        if (isnan(logits[i])) nan_l++;
    }
    printf("Text forward: %.3fs (%d layers, %d tokens)\n", t_text, model.n_layers, n_merged);
    printf("Logit range: [%.4f, %.4f] NaN=%d\n", min_l, max_l, nan_l);

    double total = now_sec() - t0;
    printf("\n=== Results ===\n");
    printf("Vision (GPU): %.3fs | Text (CPU): %.3fs | Total: %.3fs\n",
           t_vision - t0, t_text, total);
    printf("Vision tokens: %d | NaN: %s\n", n_merged, nan_c == 0 ? "YES ✓" : "NO ✗");

    // Cleanup
    free(pixels); free(h_hidden); free(h_vit_out); free(vit_embd); free(logits);
    cudaFree(d_x); cudaFree(d_out); cudaFree(d_scratch);
    for (int l = 0; l < V_N_LAYERS; l++) {
        cudaFree(d_ln1_w[l]); cudaFree(d_ln1_b[l]);
        cudaFree(d_ln2_w[l]); cudaFree(d_ln2_b[l]);
        gpu_vision_free_layer(&gpu_layers[l]);
    }
    cudaFree(d_post_ln_w); cudaFree(d_post_ln_b);
    vision_encoder_free(&enc);
    wubu_model_free(&model);
    cudaStreamDestroy(stream);
    cublasDestroy(cublas_h);

    return nan_c > 0 || nan_l > 0 ? 1 : 0;
}
