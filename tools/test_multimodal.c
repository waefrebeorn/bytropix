/*
 * test_multimodal.c -- CC01-CC07 verification.
 */
#include "wubu_imgenc.h"
#include "wubu_audio.h"
#include "wubu_mm_align.h"
#include "wubu_mm_adapter.h"
#include "wubu_mm_kv.h"
#include <stdio.h>
#include <math.h>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fails++; printf("FAIL: %s\n", msg); } \
    else printf("  ok: %s\n", msg); \
} while(0)

int main() {
    /* CC01: Vision encoder */
    printf("=== CC01: Vision Encoder (ViT patch embedding) ===\n");
    wubu_imgenc_t v;
    CHECK(wubu_imgenc_init(&v, 12345) == 0, "vision init");
    /* 64×64×3 image (grayscale ramp for determinism) */
    float img[64 * 64 * 3];
    for (int i = 0; i < 64 * 64 * 3; i++) img[i] = (float)(i % 256) / 255.0f;
    float vision_tokens[WUBU_IMGENC_N_TOKENS * WUBU_IMGENC_EMBED_DIM];
    CHECK(wubu_imgenc_encode(&v, img, vision_tokens) == 0, "vision encode");
    /* Verify output: CLS token + 64 patches, all 128-d */
    CHECK(WUBU_IMGENC_N_TOKENS == 65, "65 tokens (CLS + 64 patches)");
    int nonzero = 0;
    for (int i = 0; i < 65 * 128; i++) if (vision_tokens[i] != 0.0f) nonzero++;
    CHECK(nonzero > 100, "vision tokens non-trivial (not all zero)");
    /* Determinism: same image → same output */
    float vision_tokens2[65 * 128];
    wubu_imgenc_encode(&v, img, vision_tokens2);
    int same = 1;
    for (int i = 0; i < 65 * 128; i++)
        if (fabs(vision_tokens[i] - vision_tokens2[i]) > 1e-5) same = 0;
    CHECK(same == 1, "vision encode is deterministic");

    /* CC02: Audio encoder */
    printf("\n=== CC02: Audio Encoder (mel-spectrogram + FFT) ===\n");
    wubu_audio_t a;
    CHECK(wubu_audio_init(&a) == 0, "audio init");
    /* 2 seconds of 440 Hz sine at 16kHz = 32000 samples */
    float pcm[32000];
    for (int i = 0; i < 32000; i++)
        pcm[i] = sinf(2.0f * (float)M_PI * 440.0f * i / 16000.0f);
    float mel_out[WUBU_AUDIO_MAX_FRAMES * WUBU_AUDIO_N_MELS];
    int n_frames = wubu_audio_encode(&a, pcm, 32000, mel_out, WUBU_AUDIO_MAX_FRAMES);
    CHECK(n_frames > 0, "audio encode produced frames (>0)");
    CHECK(n_frames <= WUBU_AUDIO_MAX_FRAMES, "frames within max");
    /* A 440 Hz tone should produce energy in mel bins (non-zero output) */
    int mel_nonzero = 0;
    for (int i = 0; i < n_frames * WUBU_AUDIO_N_MELS; i++)
        if (mel_out[i] != 0.0f) mel_nonzero++;
    CHECK(mel_nonzero > 0, "mel spectrogram has non-zero output for 440Hz tone");

    /* CC03: Cross-modal alignment */
    printf("\n=== CC03: Cross-Modal Alignment ===\n");
    wubu_mm_align_t align;
    CHECK(wubu_mm_align_init(&align, 4242) == 0, "mm align init");
    float vision_aligned[65 * WUBU_MM_TEXT_DIM];
    CHECK(wubu_mm_align_vision(&align, vision_tokens, vision_aligned) == 0, "vision aligned to text space");
    int aligned_nonzero = 0;
    for (int i = 0; i < 65 * WUBU_MM_TEXT_DIM; i++)
        if (fabs(vision_aligned[i]) > 1e-6) aligned_nonzero++;
    CHECK(aligned_nonzero > 100, "aligned vision embeddings non-trivial");
    /* Audio alignment */
    float audio_aligned[WUBU_AUDIO_MAX_FRAMES * WUBU_MM_TEXT_DIM];
    CHECK(wubu_mm_align_audio(&align, mel_out, n_frames, audio_aligned) == 0, "audio aligned to text space");

    /* CC04 + CC06: Multimodal adapter (align → quantize → token IDs) */
    printf("\n=== CC04/CC06: Multimodal Adapter + Token Pipeline ===\n");
    wubu_mmvocab_t vocab;
    CHECK(wubu_mmvocab_init(&vocab, 7) == 0, "visual vocab init");
    wubu_mm_adapter_result_t result;
    CHECK(wubu_mm_adapter_run(&align, &vocab, vision_tokens, &result) == 0, "mm adapter run");
    CHECK(result.n_vision_tokens == 65, "adapter produced 65 vision token IDs");
    int distinct_ids = 0;
    for (int i = 1; i < 65; i++)
        if (result.vision_tok_ids[i] != result.vision_tok_ids[0]) distinct_ids++;
    CHECK(distinct_ids > 0, "vision token IDs are diverse (not all same)");

    /* CC05: Positional KV integration */
    printf("\n=== CC05: KV Integration ===\n");
    wubu_mm_kv_prefix_t prefix;
    CHECK(wubu_mm_kv_assemble(&result, NULL, 0, &prefix) == 65, "assembled 65 vision tokens into prefix");
    CHECK(prefix.n_tokens == 65, "prefix has 65 tokens");
    CHECK(wubu_mm_kv_safe(&prefix, 2048) == 1, "prefix is safe within 2048 ctx (no EAMM)");
    CHECK(wubu_mm_kv_safe(&prefix, 128) == 0, "prefix unsafe within 128 ctx (room for <256 text)");
    /* With audio */
    int audio_ids[10] = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
    int audio_tok_ids[10];
    /* Quantize audio tokens too */
    CHECK(wubu_mmvocab_quantize(&vocab, audio_aligned, n_frames < 10 ? n_frames : 10, audio_tok_ids) == 0,
          "audio quantized to token IDs");
    int na = n_frames < 10 ? n_frames : 10;
    wubu_mm_kv_prefix_t prefix2;
    int total = wubu_mm_kv_assemble(&result, audio_tok_ids, na, &prefix2);
    CHECK(total == 65 + na, "prefix = 65 vision + audio tokens");
    CHECK(wubu_mm_kv_safe(&prefix2, 4096) == 1, "vision+audio prefix safe within 4096");

    /* CC07: End-to-end multimodal → token IDs → safe for decode */
    printf("\n=== CC07: Integration + Safety ===\n");
    /* Full pipeline: raw image → vision tokens → aligned → quantized → prefix */
    CHECK(result.n_vision_tokens == 65 && prefix.n_tokens == 65, "end-to-end vision pipeline complete");
    CHECK(wubu_mm_kv_safe(&prefix, 512000) == 1, "prefix safe at 512K ctx (no EAMM)");

    if (fails > 0) {
        printf("\n%d TEST(S) FAILED\n", fails);
        return 1;
    }
    printf("\nALL MULTIMODAL TESTS PASSED\n");
    return 0;
}
