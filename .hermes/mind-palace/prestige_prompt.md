# WuBuText AI — Prestige Agent Prompt

Paste this as your FIRST message to the fresh Hermes CLI session.
It sets up the mind palace loop and the forward-motion process.

---

```
You are working on the WuBuText AI project.

## Boot Sequence

1. Load the `wubu-mind-palace` skill with skill_view(name='wubu-mind-palace')
2. Read `.hermes/mind-palace/README.md` — this is your navigation table of contents
3. Read `.hermes/mind-palace/plans/master_impl_plan_v2.md` — your implementation plan
4. Read `.hermes/mind-palace/plans/devils_advocate_v2.md` — risks to watch for
5. Read `.hermes/mind-palace/core/math/wubu_math_optimization_roadmap.md` — the math optimization path
6. Read `.hermes/mind-palace/timeline/README.md` — where we are in the schedule

Working directory: /home/wubu/bytropix

Model weights: /mnt/wslg/distro/models/ (Qwen3.6-35B-A3B-UD-IQ2_M.gguf is the target)

## Your Job

Execute the next step from master_impl_plan_v2.md. Follow the phase ordering:
  Phase 0 → Phase 2 → Phase 3 → Phase 4 → Phase 5
  Tokenizer (3.0) runs in parallel with Phase 2.
  CUDA kernels (Phase 6) run alongside everything.

After each step:
1. Update the plan document's progress markers
2. Save any new knowledge to memory or the wubu-mind-palace skill
3. Tell me the NEXT step to do (no extra summary)

## The Loop

For each session:
  read_plan → find_current_step → execute_it → mark_done → report_next

Don't ask what to do next. Just do the next thing and tell me what you did.
If blocked, say "BLOCKED on [reason] — [what you need from me to unblock]".
If a sub-step is done, check it off in the plan doc.
If you discover something that changes the plan or risks, update the relevant doc.

## Key Conventions

- All code is pure C. Python is for prototyping/analysis only.
- English only — no Chinese/Japanese/Korean.
- Phase 1 embeddings are already extracted: data/qwen36_embeddings_c.bin (2.03GB)
- R=0.956 for Poincaré ball radius (3 × mean_norm = 0.956)
- GGUF tensor names matter — they match what's in the GGUF file, not the model card
- Attention is SSM (Mamba2-style), not DeltaNet — read llama.cpp source to confirm tensor splits
- Toroidal optimizer (g % 2π) from baseline C code is WRONG for Poincaré — use RSGD instead
- RSGD = log_map(w,R) → subtract(lr*g) → exp_map(R)
```
