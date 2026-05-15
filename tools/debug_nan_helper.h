// Add this after #include "wubu_tst.h" in train_integrated.c
// Then sprinkle in the forward loop:

static int debug_nan_check(const float *d_buf, int n, const char *label, cudaStream_t stream) {
    static float *host = NULL;
    if(!host) host = (float*)malloc(65536 * sizeof(float)); // 256K temp
    int check = n < 32768 ? n : 32768;
    cudaMemcpyAsync(host, d_buf, check * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int nan_cnt = 0, inf_cnt = 0, big_cnt = 0;
    float max_abs = 0.0f;
    int max_idx = -1;
    for(int i = 0; i < check; i++) {
        float v = host[i];
        if(isnan(v)) nan_cnt++;
        if(isinf(v)) inf_cnt++;
        float a = fabsf(v);
        if(a > max_abs) { max_abs = a; max_idx = i; }
        if(a > 1e10f) big_cnt++;
    }
    if(nan_cnt || inf_cnt || big_cnt)
        fprintf(stderr, "  ⚠ %s: nan=%d inf=%d big(>1e10)=%d max=%.2e idx=%d (of %d sampled)\n",
                label, nan_cnt, inf_cnt, big_cnt, max_abs, max_idx, check);
    else
        fprintf(stderr, "  ✓ %s: max=%.2e\n", label, max_abs);
    return nan_cnt + inf_cnt;
}

// Check: after SSM forward (line ~338), add:
//   if(l<2) debug_nan_check(d_out, fwd_N*D_MODEL, "SSM_out", stream);
// Check: after Saxpy (line ~358), add:
//   if(l<2) debug_nan_check(d_cur, fwd_N*D_MODEL, "d_cur_after_ssm", stream);
// Check: after RMSNorm (line ~361), add:
//   if(l<2) debug_nan_check(d_np, fwd_N*D_MODEL, "post_rms", stream);
