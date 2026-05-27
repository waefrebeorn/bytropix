── bytropix ──

read workflow-parity → battleship → state → plan

workflow: ref→dump→cos-sim→locate→patch→verify→push→loop
tools: dump_ref, gen_text_cpu, layer_cos_sim, check_logits.py, py_compare_logits.py

read documents every loop. update them after every cell. never stop.

no questions. no choices. no stopping. zero delegation.

── ROOMS ──
palace: ~/bytropix/.hermes/mind-palace/
vault:  ~/bytropix/vault/
battle: ~/bytropix/.hermes/mind-palace/bytropix-300-gap-battleship.md
state:  ~/bytropix/.hermes/mind-palace/state.md
plan:   ~/bytropix/.hermes/mind-palace/plan.md
wf:     ~/bytropix/.hermes/mind-palace/workflow-parity.md

── HERMES TEST ──
tools/test-512k-suite.sh
tools/test-hermes-headless.sh
tools/test-hermes-integration.sh

── MEMORY DIRECTION ──
vault insight → write vault/[topic].md + memory target:memory content:"vault vault/[topic].md — one-line"
palace update → memory target:memory:"mind palace mind-palace/[path] — one-line"
discovery → memory target:memory:"bytropix [learned fact]"
preference → memory target:user:"wubu prefers [preference]"

── BUILD ──
make gen_text_cpu  (CPU-only inference)
make dump_ref      (reference comparison, needs llama.cpp headers/libs)

── EXECUTE ──
