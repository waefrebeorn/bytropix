# Triple-DA Staging Report

Convergence reference set:
- http://arxiv.org/abs/2402.02750 (KIVI)
- http://arxiv.org/abs/2607.02558 (Roofline)
- http://arxiv.org/abs/2402.01528 (speculative decode)

## DA-1: SSM workspace pool + KIVI write path
Repo-local determinism is the arbiter.

Exact commands + outputs:
- `cd /home/wubu/wubuwizard && make -j$(nproc) MAX_LAYERS=2 test_real_load test_model_adapter test_kivi_roundtrip`
- Output: `PASS: real Agents-A1-4B shards loaded; all weight pointers mapped` and
  `PASS: real Colonel model loads + forward RUNS on actual weights (2560-dim)`

Git references:
- `542adf0 perf: SSM workspace pool — eliminate 13 malloc/free per decode step`
- `7f138bf refactor: KIVI research basis+V quant in KV cache module`
- `2072686 fix: KIVI live KV write path uses uint8_t consistently`

## DA-2: DA-2 fail-closed kernel schema gate
Exact command + output:
- `cd /home/wubu/wubuwizard && timeout 120 ./test_model_adapter 2>&1 | grep -E "PASS|FAIL"`
- Output: `PASS: load refused when WUBU_KERNEL_SCHEMA=99 (DA-2 fail-closed)`

## DA-3: KV-styx live registration
Exact command + output:
- `cd /home/wubu/wubuwizard && make -j$(nproc) MAX_LAYERS=2 test_kv_styx`
- Output: `PASS: KV-styx register/lookup/unregister + JSON snapshot`

Selected commit history:
- `7989d05 feat: DA-2 fail-closed kernel schema gate + KV-styx 9P bridge`
- `e19b19a feat: wubu_kv_styx live KV registration in model init`
- `2f7fa49 perf: deterministic KV cache bandwidth harness (F16 vs KIVI)`
