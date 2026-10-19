# solidcv — the eyes only need the patches that moved

**Seed:** [solidsf-inc/solidcv](https://github.com/solidsf-inc/solidcv) — "Pure-Rust
OpenCV replacement & visual accelerator for LLMs, VLMs and 3D spatial computing",
Apache-2.0, by Andy Norris (solidSF Inc.).

**State at the time of reading:** public for ~3 hours, **2 commits, 8 stars, 0
releases**. The README publishes a 14-stage benchmark suite (geometric mean
12.11× vs OpenCV 5.0) measured on Apple Silicon. None of it was reproduced here,
and this note treats the numbers as claims, not facts.

## What is worth taking

One technique, and it is the one that matters for an OS with eyes:

> A VLM does not need to re-encode a frame whose content did not change.
> Split the image into a patch grid, compare with the previous frame, and send
> only the active patches while reusing KV for the rest.

The crate calls it `prune_vlm_patch_tokens`; the mechanism is a bitwise
comparison over a patch grid, which is exactly the kind of thing this project
should own in C11 rather than import as a Rust dependency.

The crate's other pillars were considered and left:

| Claim | Why it is not a lilypad today |
|---|---|
| Zobrist octree / Voxelean 3D topology | interesting, but our worldsim has no 3D volume pipeline to attach it to |
| (3+1)D worldtubes + RealityComber | needs metric camera calibration we do not have |
| 1-bit SAE circuit matching | a real idea, but the residual-stream hook does not exist yet in STAR |
| `uplift` mass-conserving zoom | genuinely good (no hallucinated pixels), but OCR is not on the critical path |
| Cloud WebSocket + "solidSF Credits" | a metered external service; against the run policy (third-party runs on our kernel, not the reverse) |

## What was built — `wubu_eye`

`include/wubu_eye.h` + `src/wubu_eye.c`, pure C11, integer-only:

- `wubu_eye_xor_popcount(a, b, n)` — exact XOR + popcount, 64 bits per step,
  tail handled per byte.
- `wubu_eye_patch_mask(prev, curr, w, h, patch, pixel_tol, threshold, mask, stats)` —
  marks the patches that changed, returns grid size, active count, changed-pixel
  count and the reuse ratio.

Two knobs are deliberately separate: **`pixel_tol`** decides whether a pixel
changed (sensor noise), **`threshold`** decides whether a patch is worth
re-encoding (a single blinking cursor should not force a 16×16 prefill).

`tools/test_eye` — **26 checks, 0 failures**:
- 128 bits differ across 16 inverted bytes; the tail byte is handled outside the
  64-bit loop
- identical 64×64 frame → 0 active, reuse 1.0
- one 16×16 patch changed → exactly 1 active, 256 changed pixels, 15/16 reuse
- one pixel: below a 2-pixel threshold, above a 1-pixel threshold
- `pixel_tol=1` ignores a delta of 1; `pixel_tol=0` counts it
- 70×50 frame → ceil grid 5×4, corner pixel lands in the last partial patch
- NULL frame / patch 0 / threshold 0 rejected

## Why this belongs to WuBu STAR

The star's photosphere is the active working set; the corona is context. Eyes
should not flood the photosphere with 4,096 tokens when 99 of them changed. The
patch mask is the valve: it converts "look at the screen" into "look at these
eleven tiles", and it composes directly with the decision ledger — the cost of a
re-encode is now a number the router can price.

The physical reading: a white dwarf's surface is mostly static, and what changes
is what we can afford to observe. `wubu_eye` is the instrument that watches the
photosphere instead of re-photographing the whole star.

## Next steps

1. Wire `wubu_eye_patch_mask` into the screen-capture path so captures emit a
   patch list plus a reuse mask, not a full frame.
2. Price the decision with `wubu_decision_ledger`: re-encode cost vs risk of
   missing a changed region; a "must-ask" class for irreversible UI actions.
3. Only then consider the 3D pillars — they need a calibrated camera, which we
   do not have.

## Honest caveats

- The crate is 3 hours old with 8 stars; the benchmark table is unverified.
- We ported one technique, not the library; no Rust, no cloud credits, no
  dependency.
- The patch grid is 2D grayscale; colour, motion vectors and sub-pixel alignment
  are not implemented.
