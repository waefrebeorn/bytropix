── bytropix ──

load skills: bytropix-cos-sim-parity, bytropix-512k-harness, bytropix-inference-server, bytropix-diagnostics, bytropix-nes-emulator

read workflow-parity → battleship → state → plan

workflow: ref→dump→cos-sim→locate→patch→verify→push→loop
tools: dump_ref, gen_text_cpu, layer_cos_sim, check_logits.py, py_compare_logits.py

skills live at ~/.hermes/skills/software-development/bytropix-*
load with: /skill bytropix-cos-sim-parity (or mention trigger words)

read documents every loop. update after every cell. never stop.
no questions. no choices. no stopping. zero delegation.

── ROOMS ──
palace: ~/bytropix/.hermes/mind-palace/
vault:  ~/bytropix/vault/
battle: ~/bytropix/.hermes/mind-palace/bytropix-300-gap-battleship.md
state:  ~/bytropix/.hermes/mind-palace/state.md
plan:   ~/bytropix/.hermes/mind-palace/plan.md
wf:     ~/bytropix/.hermes/mind-palace/workflow-parity.md

── HERMES TEST ──
tools/test-512k-suite.sh         # 6 tests: KV, attn, memory, RoPE, NES
tools/test-hermes-headless.sh    # 6 tests: server→chat→stream→agent→vault→NES
tools/test-hermes-integration.sh # 9 tests: binary→server→endpoints→format
tools/serve_local.py             # REAL inference (NOT proxy)
tools/start-bytropix-server.sh   # server startup

── MEMORY DIRECTION ──
vault insight → write vault/[topic].md + memory target:memory content:"vault vault/[topic].md — one-line"
palace update → memory target:memory:"mind palace mind-palace/[path] — one-line"
discovery → memory target:memory:"bytropix [learned fact]"
preference → memory target:user:"wubu prefers [preference]"

── BUILD ──
make gen_text_cpu  (CPU-only inference)
make dump_ref      (reference comparison, needs llama.cpp headers/libs)

── REMAINING GAPS ──
dump_ref runtime error — llama_model_load_from_file needs new API fix
run-harness.sh still uses proxy (inference-server.py) — patch to serve_local.py
NES emulator PPU on test pattern — needs proper tile/nametable + iNES loader
test-hermes-headless.sh uses proxy sandbox — update for real local mode

── EXECUTE ──
