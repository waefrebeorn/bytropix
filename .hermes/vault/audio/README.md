# Vault: Audio — WubuSynth Galactic Core Synthesizer
#
## Location: `AUDIO/wubusynth/`

### Files
- `wubusynth.py` — Main synthesizer: acoustic compander, harmonic exciter, galactic core stereo widening
- `vhf_audio.py` — VHF radio processing chain
- `audio_compressor.py` — Audio compression with learned codec (EnCodec-based)
- `vhf_tool.py` — VHF processing CLI tool
- `vhf_demos.py` — Demo generation scripts

### Architecture
sed adversarial audio synthesis uses:
    14|- EnCodec backbone for tokenization
    15|- Galactic core processing for harmonic enhancement
    16|- VHF radio chain for band-limited processing
    17|

---

*Part of the WuBuText AI project. See [Project Overview](../../README.md) and [Presentation Layer](../presentation/README.md) for navigation.*
