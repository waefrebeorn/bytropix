# Multimodal Grounding: Vision + Audio + Text Integration — 7-hop KB sweep
## CC axis: bridging non-text inputs into the text decode path (at home, C11)

> Each stone seeds the next hop. Target: map the multimodal input substrate
> that WuBuOS lacks — turning raw pixels/audio into KV-cache-positioned
> embeddings that the existing attention + causal reasoning can reason over.

## Hop 1: Vision encoder (ViT patch embedding from scratch)
ViT: image 224×224×3, patch 16×16 → 196 patches. Each patch flattened to 768-d,
projected via learned linear → embedding. Plus CLS token (1×768) prepended → 197 tokens.
No conv, no positional CNN — pure patch + linear projection + self-attention.
At home: we need a *tiny* ViT (e.g. 224×224 or 64×64, patch 8, dim 128) that
produces embeddings usable by the existing text attention stack. Pure C11,
no external deps (implement matmul, layer-norm, gelu, softmax in C).

## Hop 2: Audio encoder (mel-spectrogram + FFT from scratch)
Audio: raw PCM 16kHz → STFT (real FFT) → power spectrogram → Mel filterbank
(hz→mel: 2595*log10(1+hz/700)) → n_mels=80 bins. Librosa-like from scratch
(reddit: plain C reimplementation exists). At home: we need a minimal
spectrogram that produces a 2D time×mel tensor → tokenized into 80-d patch
vectors, matching the vision token dimension for cross-modal alignment.

## Hop 3: Cross-modal alignment (CLIP contrastive + SigLIP)
CLIP: image encoder + text encoder → shared latent space. Contrastive loss:
L = -log( exp(sim(I,T)/t) / sum_j exp(sim(I,T_j)/t) ). SigLIP replaces
sigmoid contrastive loss with a faster, noise-resilient variant.
At home: we don't have trained weights, but we can implement the *alignment
mechanism*: a cross-modal adapter that projects vision/audio tokens into the
text embedding space (dim 512 or 768) using learned or random projection
matrices. The adapter learns to map non-text features into the same space
the gen_text model already understands.

## Hop 4: Multimodal adapter (cross-attention projection)
The adapter: vision_tokens (N_v × D_v) → Linear(D_v, D_text) → vision_embeds (N_v × D_text).
Same for audio. Then cross-attention: text query attends to vision/audio keys.
Or simpler: concatenation + learned gating. BLIP-2 uses Q-Former (cross-attention
bottleneck). At home: we implement a learnable linear projection + additive bias
that maps vision embeddings into the model's embedding space, then injects
them into the KV cache at position 0 (before text tokens), so the model
"sees" the image context.

## Hop 5: Positional integration into KV cache
The embeddings must be injected into the KV cache at the right position.
Vision: 197 tokens (CLS + 196 patches) → 197 KV positions. Audio: variable
time frames → N_a KV positions. At home: WuBuOS gen_text takes a prompt string.
We extend the token pipeline: image→patches→tokens→KV-cache prefix, then
append text tokens. The vision tokens occupy KV positions [0, 197), text
occupies [197, ...). This requires no model changes — just prepending
embedding IDs before tokenization.

## Hop 6: Multimodal token pipeline (image/audio → token IDs)
Full pipeline: raw PNG/encoded audio → pixel unpack → patch embedding →
matmul projection → softmax over vocab → pseudo-token IDs → feed to gen_text.
At home: we generate pseudo-token IDs by nearest-neighbor lookup from
patch embeddings against a small "visual vocab" (learned centroids or
random anchors). This produces real token IDs the model can attend to,
turning an image into a ~197-token prefix.

## Hop 7: Integration with decode path + safety
1. PNG decode (minimal, C11 — or raw pixel buffer from filesystem)
2. → wubu_vision.c: patch embed + ViT encoder → 197 × D_text
3. → wubu_audio.c: PCM → mel-spectrogram → tokenize → N_a × D_text
4. → wubu_mm_adapter.c: project + align → text embedding space
5. → inject into KV cache prefix (position 0)
6. → gen_text attends over vision/audio context
7. Safety: vision tokens must pass loopguard (no EAMM at 512K ctx)
8. → reasoning substrate (causal, symbolic, vector search) can now
   reason over visual/auditory grounding

## Gap mapping
- CC01 Vision encoder (ViT patch embedding from scratch) `wired` (wubu_vision.c)
- CC02 Audio encoder (mel-spectrogram + real FFT) `wired` (wubu_audio.c)
- CC03 Cross-modal alignment (CLIP-style projection) `wired` (wubu_mm_align.c)
- CC04 Multimodal adapter (cross-attention → KV cache) `wired` (wubu_mm_adapter.c)
- CC05 Positional KV integration (prepend embeddings) `wired` (wubu_mm_kv.c)
- CC06 Multimodal token pipeline (image→pseudo-tokens) `wired` (wubu_mm_pipe.c)
- CC07 Integration + safety gate → decode path `wired` (test_multimodal)
- CC08 End-to-end multimodal gen_text `open` (research: needs visual vocab)
