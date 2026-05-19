#ifdef __AVX2__
void ggml_vec_dot_iq3_xxs_q8_K_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0); assert(nrc == 1); UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
    const block_iq3_xxs * GGML_RESTRICT x = vx;
    const block_q8_K   * GGML_RESTRICT y = vy;
    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        __m256i acc8 = _mm256_setzero_si256();

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            uint32_t aux32;
            memcpy(&aux32, gas, sizeof(uint32_t)); gas += sizeof(uint32_t);
            const int32_t ls = 2*(int)(aux32 >> 28) + 1;
            __m256i scale_v = _mm256_set1_epi32(ls);

            // Process 4 pairs (8 grid entries) = 32 elements
            // Each grid entry -> 4 bytes via lookup
            for (int l = 0; l < 4; l += 2) {
                // Load 2 grid indices
                uint8_t g0 = q3[2*l+0], g1 = q3[2*l+1];
                uint8_t g2 = q3[2*l+2], g3 = q3[2*l+3];

                // Gather 4 grid values (each 4 bytes = 4 pixel values)
                // Grid entries are uint32_t, each byte is a pixel
                uint32_t idxs[4] = {g0, g1, g2, g3};
                __m128i grid_idx = _mm_loadu_si128((const __m128i*)idxs);
                // grid_idx contains {g0, g1, g2, g3} as 32-bit integers

                // Gather from iq3xxs_grid (uint32_t[256])
                __m128i gv0 = _mm_i32gather_epi32((const int*)iq3xxs_grid, grid_idx, 4);
                // gv0 = {grid[g0], grid[g1], grid[g2], grid[g3]} each as uint32

                // Load signs
                const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                __m256i sign_vals;
                {
                    // Create sign mask from byte bits
                    uint32_t sm[8];
                    for (int j = 0; j < 8; j++) sm[j] = (signs & kmask_iq2xs[j]) ? (uint32_t)-1 : 0;
                    sign_vals = _mm256_loadu_si256((const __m256i*)sm);
                }

                // Load 8 q8 values
                __m256i q8v = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q8)));

                // Dequant grid values (each uint32 -> 4 bytes)
                // Extract individual bytes from the grid entries
                uint8_t grid_bytes[16];
                memcpy(grid_bytes, &gv0, 16); // 4 uint32 = 16 bytes
                __m256i grid_vals = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)grid_bytes));

                // Apply signs: if sign bit set, negate
                // sign_vals = all-1s for negative, all-0s for positive
                // XOR: x ^ -1 = ~x, so negate needs (x ^ -1) + 1 = -x
                // But we can just multiply: grid * sign (+1 or -1)
                __m256i signed_grid = _mm256_sign_epi8(
                    _mm256_castsi256_si128(grid_vals),
                    _mm256_castsi256_si128(sign_vals));
                
                // Dot product: signed_grid * q8v, accumulate into acc8
                __m256i prod = _mm256_mullo_epi32(signed_grid, q8v);
                acc8 = _mm256_add_epi32(acc8, _mm256_mullo_epi32(prod, scale_v));

                q8 += 8;
            }
            q3 += 8;
        }

        // Horizontal sum of acc8
        __m128i lo = _mm256_castsi256_si128(acc8);
        __m128i hi = _mm256_extracti128_si256(acc8, 1);
        __m128i sum128 = _mm_add_epi32(lo, hi);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        sum128 = _mm_hadd_epi32(sum128, sum128);
        int32_t sum_val = _mm_cvtsi128_si32(sum128);
        sumf += d * sum_val;
    }
    *s = 0.25f * sumf;
}
#endif // __AVX2__
