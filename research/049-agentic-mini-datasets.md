# 049 — Agentic mini-datasets: the OS-backbone tiers

> Status: `closed` (data fetched + packed). Date: 2026-08-04.
> Maps to: THEME AG (the wubuos agentic-corpus bank, 1000 gaps) +
> the WuBu-35M agentic role as the WuBuOS backbone.

## Why

WuBu-35M is the brain inside the WuBuOS body — it must do TERMINAL
interaction (the shell), TOOL CALLING (the VSL personalities), and know
the ANCIENT subsystems it hosts (CP/M, Classic Mac, DOS, XNU). The big
labs' agentic data is huge; we take the light, high-signal slices.

## The pack

`/home/wubu/models/corpus/agentic/wubu-agentic-pack.jsonl`
**1,392 conversations, 1.0 MB**, deterministic seed 48.

| Tier | Source | Count | What it teaches |
|---|---|---|---|
| terminal | princeton-nlp/InterCode nl2bash | 224 | NL -> bash command pairs |
| OS tasks | THUDM/AgentBench os_interaction | 1,000 | file/cron/process sysadmin tasks with explanations |
| ancient | generated from wubuos source | 168 | toast-OS syscall Q/A (the personalities) |

## The ancient-subsystem generator (unique to us)

`/home/wubu/research-ponds-work/tools/gen_ancient_corpus.py` parses the
DISPATCH TABLES in wubuos's own VSL layer — the ground truth of every
syscall the model will service:

- CP/M BDOS: 24 syscalls (`vsl_syscall_cpm.c`)
- Classic Mac 68K: 17 traps (`vsl_syscall_macclassic.c`)
- DOS INT 21h/10h/16h: 43 cases (`wubu_dos_emu_int.c`)
- XNU macOS BSD: 84 syscalls (`vsl_syscall_mac_bsd.c`)

Output: `ancient-subsystem.jsonl` (168 Q/A). This data EXISTS nowhere
else — it's the exact tables our emulator implements, so the model learns
the real subsystem it backs.

## Build tools (all in /home/wubu/research-ponds-work/tools/)

- `fetch_sft_data.py` — HF downloader (gsm8k/ultrachat/slimorca)
- `build_sft_pack.py` — Tier 1 SFT pack (19,473 convos)
- `gen_ancient_corpus.py` — ancient-subsystem generator
- `build_agentic_pack.py` — merges nl2bash+agentbench+ancient (1,392)

## Next step

Tokenize wubu-agentic-pack.jsonl (conversation roles -> special tokens)
and add as the Tier-2 mix in the SFT cold-start run. The wubuos
agentic-corpus bank (docs/compendium/04-roadmap/agentic-corpus-bank.md,
1000 gaps) is the roadmap for the FULL agentic data synthesis wave —
these mini-datasets are the seed for that.
