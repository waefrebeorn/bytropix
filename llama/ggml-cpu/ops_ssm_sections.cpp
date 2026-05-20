// ggml_compute_forward_ssm_conv

static void ggml_compute_forward_ssm_conv_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // conv_x
    const ggml_tensor * src1 = dst->src[1]; // conv1d.weight

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc  = src1->ne[0]; // d_conv
    const int ncs = src0->ne[0]; // d_conv - 1 + n_t
    const int nr  = src0->ne[1]; // d_inner
    const int n_t =  dst->ne[1]; // tokens per sequence
    const int n_s =  dst->ne[2]; // number of sequences in the batch

    GGML_ASSERT( dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);
    const int ir  = ir1 - ir0;

    for (int i3 = 0; i3 < n_s; ++i3) {
        for (int i2 = 0; i2 < n_t; ++i2) {
            // {d_conv - 1 + n_t, d_inner, n_seqs}
            // sliding window
            const float * s = (const float *) ((const char *) src0->data + ir0*(src0->nb[1]) + i2*(src0->nb[0]) + i3*(src0->nb[2])); // {d_conv, d_inner, n_s}
            const float * c = (const float *) ((const char *) src1->data + ir0*(src1->nb[1])); // {d_conv, d_inner}
            float * x = (float *) ((char *) dst->data + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2])); // {d_inner, n_t, n_s}

            // TODO: transpose the output for smaller strides for big batches?
            // d_inner
            for (int i1 = 0; i1 < ir; ++i1) {
                // rowwise dot product
                // NOTE: not using ggml_vec_dot_f32, because its sum is in double precision
                float sumf = 0.0f;

                // d_conv
                for (int i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
                }
                x[i1] = sumf;
            }
        }
    }
}

void ggml_compute_forward_ssm_conv(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_conv_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_scan

static void ggml_compute_forward_ssm_scan_f32(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // s  {d_state, dim, n_head, n_seqs+}
    const ggml_tensor * src1 = dst->src[1]; // x  {dim, n_head, n_seq_tokens, n_seqs}
    const ggml_tensor * src2 = dst->src[2]; // dt {n_head, n_seq_tokens, n_seqs}
    const ggml_tensor * src3 = dst->src[3]; // A  {d_state, n_head} or {1, n_head}
    const ggml_tensor * src4 = dst->src[4]; // B  {d_state, n_group, n_seq_tokens, n_seqs}
    const ggml_tensor * src5 = dst->src[5]; // C  {d_state, n_group, n_seq_tokens, n_seqs}
    const ggml_tensor * src6 = dst->src[6]; // ids {n_seqs}

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nc = src0->ne[0]; // d_state
    const int64_t nr = src0->ne[1]; // dim
    const int64_t nh = src1->ne[1]; // n_head
    const int64_t ng = src4->ne[1];
    const int64_t nt = src1->ne[2]; // number of tokens per sequence
    const int64_t ns = src1->ne[3]; // number of sequences in the batch

    // can't use ggml_nbytes because src1 is not necessarily contiguous
    const int64_t s_off = ggml_nelements(src1) * ggml_element_size(src1);

    GGML_ASSERT(ggml_nelements(src1) + nc*nr*nh*ns == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src2->nb[0] == sizeof(float));
    GGML_ASSERT(src3->nb[0] == sizeof(float));
    GGML_ASSERT(src4->nb[0] == sizeof(float));
    GGML_ASSERT(src5->nb[0] == sizeof(float));
    GGML_ASSERT(src6->nb[0] == sizeof(int32_t));
    GGML_ASSERT(nh % ng == 0);

    // heads per thread
    const int dh = (nh + nth - 1)/nth;

    // head range for this thread
    const int ih0 = dh*ith;
    const int ih1 = MIN(ih0 + dh, nh);

    const int32_t * ids = (const int32_t *) src6->data;

    for (int i3 = 0; i3 < ns; ++i3) {
        const float * s0 = (const float *) ((const char *) src0->data + ids[i3]*(src0->nb[3])); // {d_state, dim, nh, ns}
              float * s  = (      float *) ((      char *) dst->data  + i3*(src0->nb[3]) + s_off); // {d_state, dim, nh, ns}

        for (int i2 = 0; i2 < nt; ++i2) {
            const float * x  = (const float *) ((const char *) src1->data + i2*(src1->nb[2]) + i3*(src1->nb[3])); // {dim, nh, nt, ns}
            const float * dt = (const float *) ((const char *) src2->data + i2*(src2->nb[1]) + i3*(src2->nb[2])); // {nh, nt, ns}
            const float * A  = (const float *) ((const char *) src3->data); // {d_state, nh} or {1, nh}
            const float * B  = (const float *) ((const char *) src4->data + i2*(src4->nb[2]) + i3*(src4->nb[3])); // {d_state, ng, nt, ns}
            const float * C  = (const float *) ((const char *) src5->data + i2*(src5->nb[2]) + i3*(src5->nb[3])); // {d_state, ng, nt, ns}
                  float * y  = (      float *) ((      char *) dst->data + i2*(nh*nr*sizeof(float)) + i3*(nt*nh*nr*sizeof(float))); // {dim, nh, nt, ns}

            if (src3->ne[0] == 1) {
                // Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop

                // n_head
                for (int h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    const float dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    const float dA = expf(dt_soft_plus * A[h]);
                    const int g = h / (nh / ng); // repeat_interleave

                    // dim
                    for (int i1 = 0; i1 < nr; ++i1) {
                        const int ii = i1 + h*nr;
                        const float x_dt = x[ii] * dt_soft_plus;
                        float sumf = 0.0f;
#if defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)
                        const int ggml_f32_epr = svcntw();
                        const int ggml_f32_step = 1 * ggml_f32_epr;

                        const int np = (nc & ~(ggml_f32_step - 1));

                        GGML_F32_VEC sum = GGML_F32_VEC_ZERO;

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        for (int i = 0; i < np; i += ggml_f32_step) {
                            // TODO: maybe unroll more?
                            for (int j = 0; j < 1; j++) {
                                GGML_F32_VEC t0 = GGML_F32_VEC_LOAD(s0 + i + j*ggml_f32_epr + ii*nc);
                                GGML_F32_VEC t1 = GGML_F32_VEC_LOAD(B + i + j*ggml_f32_epr + g*nc);
                                GGML_F32_VEC t2 = GGML_F32_VEC_LOAD(C + i + j*ggml_f32_epr + g*nc);

                                t0 = GGML_F32_VEC_MUL(t0, adA);
                                t1 = GGML_F32_VEC_MUL(t1, axdt);

                                t0 = GGML_F32_VEC_ADD(t0, t1);

                                sum = GGML_F32_VEC_FMA(sum, t0, t2);

                                GGML_F32_VEC_STORE(s + i + j*ggml_f32_epr + ii*nc, t0);
                            }
                        }

                        sumf = GGML_F32xt_REDUCE_ONE(sum);
    #elif defined(__riscv_v_intrinsic)
                        // todo: RVV implementation
                        const int np = 0;
    #else
                        const int np = (nc & ~(GGML_F32_STEP - 1));

                        GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        GGML_F32_VEC ax[GGML_F32_ARR];
                        GGML_F32_VEC ay[GGML_F32_ARR];
                        GGML_F32_VEC az[GGML_F32_ARR];

                        for (int i = 0; i < np; i += GGML_F32_STEP) {
                            for (int j = 0; j < GGML_F32_ARR; j++) {
                                ax[j] = GGML_F32_VEC_LOAD(s0 + i + j*GGML_F32_EPR + ii*nc);
                                ay[j] = GGML_F32_VEC_LOAD(B + i + j*GGML_F32_EPR + g*nc);
                                az[j] = GGML_F32_VEC_LOAD(C + i + j*GGML_F32_EPR + g*nc);

                                ax[j] = GGML_F32_VEC_MUL(ax[j], adA);
                                ay[j] = GGML_F32_VEC_MUL(ay[j], axdt);

                                ax[j] = GGML_F32_VEC_ADD(ax[j], ay[j]);

                                sum[j] = GGML_F32_VEC_FMA(sum[j], ax[j], az[j]);

                                GGML_F32_VEC_STORE(s + i + j*GGML_F32_EPR + ii*nc, ax[j]);
                            }
                        }

                        // reduce sum0..sum3 to sum0
                        GGML_F32_VEC_REDUCE(sumf, sum);
    #endif
#else
                        const int np = 0;
#endif
                        // d_state
                        for (int i0 = np; i0 < nc; ++i0) {
                            const int i = i0 + ii*nc;
                            const int ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            const float state = (s0[i] * dA) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
                    }
                }
            } else {
                // Mamba-1 has an element-wise decay factor for the states

                // n_head
                for (int h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    const float dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    const int g = h / (nh / ng); // repeat_interleave

                    // dim
                    for (int i1 = 0; i1 < nr; ++i1) {
                        const int ii = i1 + h*nr;
                        const float x_dt = x[ii] * dt_soft_plus;
#if defined(__ARM_FEATURE_SVE)
                        svfloat32_t vx_dt = GGML_F32_VEC_SET1(x_dt);
                        svfloat32_t vdt_soft_plus = GGML_F32_VEC_SET1(dt_soft_plus);
                        svfloat32_t r1_vector = GGML_F32_VEC_ZERO;

                        // d_state
                        // TODO: what happens when (d_state % svcntw()) != 0?
                        for (int64_t k = 0; k < nc; k += svcntw()) {
                            svfloat32_t vA = GGML_F32_VEC_LOAD(&A[h*nc + k]);
                            svfloat32_t vB = GGML_F32_VEC_LOAD(&B[k + g*nc]);
                            svfloat32_t vC = GGML_F32_VEC_LOAD(&C[k + g*nc]);
                            svfloat32_t vs0 = GGML_F32_VEC_LOAD(&s0[ii*nc + k]);

                            svfloat32_t t1 = GGML_F32_VEC_MUL(vdt_soft_plus, vA);
                            t1 = exp_ps_sve(svptrue_b32(), t1);
                            svfloat32_t t2 = GGML_F32_VEC_MUL(vx_dt, vB);

                            vs0 = GGML_F32_VEC_FMA(t2, vs0, t1);
                            r1_vector = GGML_F32_VEC_ADD(GGML_F32_VEC_MUL(vs0, vC), r1_vector);

                            GGML_F32_VEC_STORE(&s[ii*nc + k], vs0);
                        }
                        y[ii] = GGML_F32xt_REDUCE_ONE(r1_vector);
#else
                        float sumf = 0.0f;
                        // NOTE: can't really use GGML_SIMD here because d_state is usually 16
                        //       and also because expf is used within the loop.
                        // d_state
                        for (int i0 = 0; i0 < nc; ++i0) {
                            const int i = i0 + ii*nc;
                            const int ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            const float state = (s0[i] * expf(dt_soft_plus * A[i0 + h*nc])) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
#endif
                    }
                }
            }
            // use the output as the source when it's not the first token-wise iteration
            s0 = s;
        }
    }
}

void ggml_compute_forward_ssm_scan(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_scan_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

