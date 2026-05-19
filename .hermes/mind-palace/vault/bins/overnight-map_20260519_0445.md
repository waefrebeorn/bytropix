# Overnight Map — Per-Layer Cos-Sim (May 19, 2026)

## Layer Dump Comparison (BOS token)
Ran `tools/layer_cos_sim` against existing /tmp/dump_layers_ref/ vs /tmp/dump_layers_our/:

```
L00-SSM: cos=0.8598 max_diff=0.1986
L01-SSM: cos=0.7464 max_diff=0.1617
L02-SSM: cos=0.9360 max_diff=0.3460
L03-GQA: cos=0.9193 max_diff=0.3877
L04-SSM: cos=0.9160 max_diff=0.4026
L05-SSM: cos=0.9107 max_diff=0.3880
L06-SSM: cos=0.9711 max_diff=0.8560
L07-GQA: cos=0.9693 max_diff=0.8165
... (gradual 0.97→0.88 through L31)
L32-SSM: cos=0.8570 max_diff=1.528
L33-SSM: cos=0.8312 max_diff=1.957
L34-SSM: cos=0.7894 max_diff=2.650
L35-GQA: cos=0.7535 max_diff=3.184
L36-SSM: cos=0.7484 max_diff=3.147
L37-SSM: cos=0.6301 max_diff=4.743
L38-SSM: cos=0.4540 max_diff=7.359
L39-GQA: cos=0.4610 max_diff=6.640
OVERALL: cos=0.8792
```

## Analysis
1. **SSM divergence starts at L0**: cos=0.86. SSM kernel produces different results.
2. **GQA also diverges**: L3 cos=0.92. Not as bad as SSM but still significant.
3. **Gradual decay**: L6-L31 (0.97→0.88) — expected for quant noise amplification.
4. **Sharp drop L32-L39**: Values diverge massively (max_diff ~7). Something systematic in late layers — could be cumulative from earlier divergence.

## Tool Created
`tools/layer_cos_sim.c` — compares ref/our layer dumps, computes per-layer cos-sim + max_diff.

## Next Steps
1. Add SSM intermediate dump to llama.cpp (Q, K, V, gate, beta, conv_out, state)
2. Dump same intermediates from bytropix via DUMP_SSM_DEBUG
3. Compare step-by-step to find exact divergence point
4. Fix the discrepancy
