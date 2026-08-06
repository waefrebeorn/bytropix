# Backup AGI — the sensory-motor periphery (2026-08-05)

The user's vision: the primary AGI stack has a **backup** that keeps the
loop alive with the smallest possible footprint. The backup is asymmetric
on purpose — neural models for the inputs (pattern recognition can't be
cheated), a physical model for the output (articulatory synthesis needs
no weights).

## The stack

| Role | Component | What it is | Why it's the right choice |
|------|-----------|-----------|--------------------------|
| 👂 ears | NVIDIA Nemotron ASR 0.6B | speech → text | 0.6B, runs on a potato, streaming-capable |
| 👁️ eyes | Zhipu GLM-OCR 0.9B | pixels → text | 0.9B, reads the world (docs, screens, photos) |
| 🧠 brain | DeepSeek V4 Flash 0731 | thinking core | the live model (deepseek-v4-flash-0731) |
| 🗣️ throat | Pink Trombone | articulatory speech synthesis | a PHYSICAL vocal tract model, not a neural TTS |

## Why Pink Trombone is the "backup throat"

Pink Trombone (Neil Thapen) is not a text-to-speech model — it is a
**2D digital waveguide model of the human vocal tract**:

- Glottis pulse source (periodic/aspirate mix, controlled by
  `glottalPulseShape` + frequency)
- Tube sections with **area functions** — tongue position/width,
  lip opening, velum (nasal coupling) all become real controllable state
- Nasal tract (parallel resonator)
- Lip radiation filter
- Renders vowels and consonants from **articulatory parameters**, not
  from text embeddings

Consequences that make it the right "backup throat":

1. **Tiny** — the original is one JS file of pure DSP (~1000 lines). A
   C11 port is a few hundred lines; no weights to host, no GPU, no
   third-party deps. Fits the "no third party if we can write it" rule.
2. **Fully embodied** — tongue, jaw, lips, velum, glottis are real
   state. The AGI must LEARN to control the throat (articulatory
   targets → trajectories), which is the same shape as WuBu's own
   philosophy: the body is a physical instrument, the brain learns to
   play it.
3. **Lossless by construction** — deterministic, no sampling variance,
   no model collapse. Render is reproducible.
4. **Ports exist** — Rust crate `pink-trombone` on lib.rs; JS original;
   a C11 port drops straight into WuBuOS `src/audio/`.

## How it maps onto the repos

- **wubuwizard** = the brain core (DeepSeek-class thinking on-device)
- **WuBuOS** = the body: ASR + OCR as input device drivers, Pink
  Trombone waveguide as an audio device driver
- The Styx/9P namespace (ADR-003 / research 061) already has the shape:
  `/dev/ears`, `/dev/eyes`, `/dev/throat` — every sense a file

```
mic → /dev/ears (ASR) → text
camera/screen → /dev/eyes (OCR) → text
text → brain (DeepSeek V4 Flash) → intent
intent → learned controller → articulatory targets → /dev/throat
     → Pink Trombone waveguide → PCM → speakers
```

## "The pink trombone learned"

The key phrase. The throat itself is physics; the *controller* is
learned: a small net maps intent → articulatory trajectories → waveguide.
Two ways to drive it:
1. **Direct articulatory control** — set tongue/lip/velum/glottis per
   frame; the AGI learns the mapping end-to-end (RL or supervised from
   real speech articulation data).
2. **Formant-target control** — higher level: target F1/F2/F3 + voicing,
   a small solver converts formant targets → tube areas (the classic
   inverse problem of articulatory synthesis).

## Next steps (when the user says "wire it in")

1. Port the Pink Trombone DSP to C11 (`wubu_throat.c` — glottis source,
   tube waveguide, nasal tract, lip radiation, `render(f32* out, n)`)
2. Register `/dev/throat` in the WuBuOS Styx namespace (write
   articulatory params in, read PCM out — file semantics per ADR-003)
3. `/dev/ears` — PCM capture + Nemotron ASR bridge (ONNX or via the
   wubuwizard safetensors bridge)
4. `/dev/eyes` — image capture + GLM-OCR bridge
5. Learned controller: small net, articulatory targets from text/phonemes

Status: `research` (design recorded 2026-08-05; nothing shipped yet).
