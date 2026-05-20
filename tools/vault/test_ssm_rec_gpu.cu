#include "gguf_reader.h"
#include "wubu_ssm.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define V_HEADS 32

// The GPU kernel declaration
__global__ void ssm_recurrence_kernel(
    float *ssm_state, const float *q_all, const float *k_all,
    const float *v_all, const float *beta, const float *gate,
    float *delta_out, float q_scale);

// CPU reference for one head
static void cpu_recurrence(float *h, const float *q, const float *k,
    const float *v, float bg, float gg, float *delta, float q_scale) {
    // Decay
    for (int i = 0; i < SSM_D_STATE; i++)
        for (int j = 0; j < SSM_D_STATE; j++)
            h[i * SSM_D_STATE + j] *= gg;
    // h @ k
    float hk[SSM_D_STATE] = {0};
    for (int i = 0; i < SSM_D_STATE; i++)
        for (int j = 0; j < SSM_D_STATE; j++)
            hk[i] += h[i * SSM_D_STATE + j] * k[j];
    // diff = v - hk
    float diff[SSM_D_STATE];
    for (int i = 0; i < SSM_D_STATE; i++) diff[i] = v[i] - hk[i];
    // State update
    for (int i = 0; i < SSM_D_STATE; i++)
        for (int j = 0; j < SSM_D_STATE; j++)
            h[i * SSM_D_STATE + j] += bg * k[i] * diff[j];
    // h @ q
    for (int i = 0; i < SSM_D_STATE; i++) {
        double sum = 0;
        for (int j = 0; j < SSM_D_STATE; j++)
            sum += h[i * SSM_D_STATE + j] * q[j];
        delta[i] = (float)sum * q_scale;
    }
}

int main() {
    srand(42);
    float q_scale = 1.0f / sqrtf(SSM_D_STATE);

    // Setup one V-head
    float h[SSM_D_STATE * SSM_D_STATE];
    float q[SSM_D_STATE], k[SSM_D_STATE], v[SSM_D_STATE];
    float delta_cpu[SSM_D_STATE];
    for (int i = 0; i < SSM_D_STATE * SSM_D_STATE; i++)
        h[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < SSM_D_STATE; i++) {
        q[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
        k[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
        v[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    }
    float bg = 0.5f, gg = expf(0.1f); // beta=0.5, gate=0.1→decay=1.105

    // Copy before CPU modifies it
    float h_cpu[SSM_D_STATE * SSM_D_STATE];
    memcpy(h_cpu, h, sizeof(h));

    // CPU
    cpu_recurrence(h_cpu, q, k, v, bg, gg, delta_cpu, q_scale);

    // GPU
    float *d_h, *d_q, *d_k, *d_v, *d_beta, *d_gate, *d_delta;
    cudaMalloc(&d_h, sizeof(h));
    cudaMalloc(&d_q, sizeof(q));
    cudaMalloc(&d_k, sizeof(k));
    cudaMalloc(&d_v, sizeof(v));
    cudaMalloc(&d_beta, 4);
    cudaMalloc(&d_gate, 4);
    cudaMalloc(&d_delta, sizeof(delta_cpu));

    cudaMemcpy(d_h, h, sizeof(h), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q, sizeof(q), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, sizeof(k), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, sizeof(v), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, &bg, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gate, &gg, 4, cudaMemcpyHostToDevice);  // hmm, gate is already decay, not log
    // Actually the kernel expects gate (log decay), so pass log(gg)
    float log_gate = logf(gg);
    cudaMemcpy(d_gate, &log_gate, 4, cudaMemcpyHostToDevice);

    float beta_arr[V_HEADS] = {0}, gate_arr[V_HEADS] = {0};
    float q_all[V_HEADS * SSM_D_STATE] = {0}, k_all[V_HEADS * SSM_D_STATE] = {0};
    float v_all[V_HEADS * SSM_D_STATE] = {0};
    beta_arr[0] = bg; gate_arr[0] = log_gate;
    memcpy(q_all, q, sizeof(q));
    memcpy(k_all, k, sizeof(k));
    memcpy(v_all, v, sizeof(v));

    float *d_q_all, *d_k_all, *d_v_all, *d_beta_arr, *d_gate_arr;
    cudaMalloc(&d_q_all, sizeof(q_all));
    cudaMalloc(&d_k_all, sizeof(k_all));
    cudaMalloc(&d_v_all, sizeof(v_all));
    cudaMalloc(&d_beta_arr, sizeof(beta_arr));
    cudaMalloc(&d_gate_arr, sizeof(gate_arr));
    cudaMemcpy(d_q_all, q_all, sizeof(q_all), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_all, k_all, sizeof(k_all), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_all, v_all, sizeof(v_all), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta_arr, beta_arr, sizeof(beta_arr), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gate_arr, gate_arr, sizeof(gate_arr), cudaMemcpyHostToDevice);

    float *d_state;
    cudaMalloc(&d_state, V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
    cudaMemset(d_state, 0, V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
    // Copy h into head 0
    cudaMemcpy(d_state, h, sizeof(h), cudaMemcpyHostToDevice);

    ssm_recurrence_kernel<<<V_HEADS, SSM_D_STATE>>>(
        d_state, d_q_all, d_k_all, d_v_all, d_beta_arr, d_gate_arr,
        d_delta, q_scale);
    cudaDeviceSynchronize();

    float delta_gpu[SSM_D_STATE];
    cudaMemcpy(delta_gpu, d_delta, sizeof(delta_gpu), cudaMemcpyDeviceToHost);

    // Compare
    double dt = 0, n1 = 0, n2 = 0, me = 0;
    for (int i = 0; i < SSM_D_STATE; i++) {
        dt += delta_cpu[i] * delta_gpu[i];
        n1 += delta_cpu[i] * delta_cpu[i];
        n2 += delta_gpu[i] * delta_gpu[i];
        double e = fabs(delta_cpu[i] - delta_gpu[i]);
        if (e > me) me = e;
    }
    printf("CPU[0..4]: %.6f %.6f %.6f %.6f %.6f\n", delta_cpu[0], delta_cpu[1], delta_cpu[2], delta_cpu[3], delta_cpu[4]);
    printf("GPU[0..4]: %.6f %.6f %.6f %.6f %.6f\n", delta_gpu[0], delta_gpu[1], delta_gpu[2], delta_gpu[3], delta_gpu[4]);
    printf("cos-sim=%.8f max_err=%.6f\n", dt/(sqrt(n1)*sqrt(n2)), me);

    cudaFree(d_h); cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_beta); cudaFree(d_gate); cudaFree(d_delta);
    cudaFree(d_q_all); cudaFree(d_k_all); cudaFree(d_v_all);
    cudaFree(d_beta_arr); cudaFree(d_gate_arr); cudaFree(d_state);
    return 0;
}
