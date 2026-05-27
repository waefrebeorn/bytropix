── bytropix: context growth penalty ──

read walkway → battleship → state → plan → vault/context-growth-penalty

workflow: read → pick → implement → verify → update docs → push → loop
tools: cat, write_file, patch, terminal, memory, vault

skills: bytropix-cos-sim-parity, bytropix-512k-harness, bytropix-inference-server, bytropix-diagnostics, bytropix-nes-emulator
load with: mention trigger words

read documents every loop. update after every cell. never stop.
no questions. no choices. no time. zero delegation.

── ROOMS ──
palace: ~/bytropix/.hermes/mind-palace/
vault:  ~/bytropix/vault/
walk:   ~/bytropix/.hermes/mind-palace/walkway.md
battle: ~/bytropix/.hermes/mind-palace/bytropix-300-gap-battleship.md
state:  ~/bytropix/.hermes/mind-palace/state.md
plan:   ~/bytropix/.hermes/mind-palace/plan.md
wf:     ~/bytropix/.hermes/mind-palace/workflow-parity.md

── FIXING CONTEXT GROWTH PENALTY ──
decode drops 1.2→0.6 tok/s as context grows (turn 2→3)
dense attention O(n²) kills short/medium context
sparse attn activates at >4K — too high for short convos
fix: lower threshold to 512+ OR optimize dense path

── HERMES TEST ──
tools/test-512k-suite.sh         # 6 tests: KV, attn, memory, RoPE, NES
tools/test-hermes-headless.sh    # 6 tests: server→chat→stream→agent→vault→NES
tools/test-hermes-integration.sh # 9 tests: binary→server→endpoints→format
tools/serve_local.py             # REAL inference (NOT proxy)
tools/start-bytropix-server.sh   # server startup

── VAULT & MEMORY DIRECTION ──
discover → write vault/[topic].md + memory target:memory content:"vault vault/[topic].md — one-line what it covers"
palace insight → memory target:memory content:"mind palace mind-palace/[path] — one-line update"
preference → memory target:user content:"wubu prefers [preference]"

── CORE DIRECTION ──
Read documents. Pick lowest undone cell. Execute. Update docs. Push. Loop.
No stopping. Every output feeds back into documents.
Read → update → read → update → never stop.

── EXECUTE ──
