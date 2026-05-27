── bytropix ──

load skills: bytropix-cos-sim-parity, bytropix-512k-harness, bytropix-inference-server, bytropix-diagnostics, bytropix-nes-emulator

read workflow-parity → battleship → state → plan → README

workflow: read → pick → update → verify → push → loop
tools: cat, write_file, patch, terminal, memory, vault

skills live at ~/.hermes/skills/software-development/bytropix-*
load with: /skill bytropix-cos-sim-parity (or mention trigger words)

read documents every loop. update after every cell. never stop.
no questions. no choices. no time. zero delegation.

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

── VAULT & MEMORY DIRECTION ──
Discovery → write vault/[topic].md + memory target:memory content:"vault vault/[topic].md — one-line"
Palace insight → memory target:memory content:"mind palace mind-palace/[path] — one-line"
Workflow learning → memory target:memory content:"bytropix [learned fact]"
Preference → memory target:user content:"wubu prefers [preference]"

── BUILD ──
make gen_text_cpu  (CPU-only inference)
make dump_ref      (reference comparison, needs llama.cpp)

── CORE DIRECTION ──
Read documents. Pick lowest undone cell in battleship. Execute. Update docs. Push. Loop.
No stopping. Every output feeds back into documents.
Read → update → read → update → never stop.

── EXECUTE ──
