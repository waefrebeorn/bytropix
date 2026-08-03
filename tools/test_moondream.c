/* test_moondream.c -- MoonDream 3 vision core: the agentic vision frontier. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wubu_moondream.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_moondream (MD01-MD10) ===\n");

    /* MD01: preprocess */
    {
        uint8_t raw[12] = { 0, 128, 255, 0, 128, 255, 255, 0, 128, 255, 0, 128 };
        wubu_image_t img;
        CHECK(wubu_md3_preprocess(raw, 2, 2, 3, &img) == 12, "preprocess");
        NEAR(img.pixels[0], 0.0f, 1e-5f);
        NEAR(img.pixels[1], 128.0f / 255.0f, 1e-4f);
        NEAR(img.pixels[2], 1.0f, 1e-5f);
        free(img.pixels);
    }

    /* MD02: encode */
    {
        float pixels[4 * 4 * 3];
        for (int i = 0; i < 4 * 4 * 3; i++) pixels[i] = (float)i / 48.0f;
        wubu_image_t img = { pixels, 4, 4, 3 };
        float tokens[4 * 4 * 4];
        CHECK(wubu_md3_encode(&img, tokens, 16) == 4, "encode 2x2 patches");
    }

    /* MD03: MoE forward */
    {
        float tokens[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        int experts[2] = { 0, 1 };
        float out[8];
        CHECK(wubu_md3_moe_forward(tokens, 2, experts, 2, out, 4) == 2, "MoE forward");
    }

    /* MD04: detect */
    {
        float encoded[56];
        for (int i = 0; i < 56; i++) encoded[i] = (i % 3 == 0) ? 0.9f : 0.1f;
        wubu_md3_result_t result = { 0 };
        int n = wubu_md3_detect(encoded, 14, "car", &result, 10);
        CHECK(n > 0, "detect found objects");
        if (result.objects) {
            CHECK(result.objects[0].confidence > 0, "confidence > 0");
            free(result.objects);
        }
    }

    /* MD05: caption */
    {
        float encoded[56];
        for (int i = 0; i < 4; i++) encoded[i] = 0.6f;  /* warm */
        for (int i = 4; i < 56; i++) encoded[i] = 0.2f;
        char caption[256];
        wubu_md3_caption(encoded, 14, caption, 256);
        CHECK(strlen(caption) > 0, "caption generated");
    }

    /* MD06: toolcall */
    {
        float encoded[16] = { 0 };
        wubu_md3_tool_t tools[1];
        int n = wubu_md3_toolcall(encoded, 4, "detect car", tools, 1);
        CHECK(n == 1, "toolcall extracted");
        CHECK(strcmp(tools[0].name, "detect") == 0, "tool name is detect");
    }

    /* MD07: end-to-end inference */
    {
        uint8_t raw[4 * 4 * 3];
        for (int i = 0; i < 4 * 4 * 3; i++) raw[i] = (uint8_t)(i * 17);
        wubu_md3_output_t out;
        CHECK(wubu_md3_infer(raw, 4, 4, 3, "describe", &out) == 0, "e2e infer");
        CHECK(out.caption != NULL, "caption produced");
        if (out.caption) { free(out.caption); }
        if (out.detections.objects) { free(out.detections.objects); }
        if (out.tools) { free(out.tools); }
    }
    {
        /* e2e detect path */
        uint8_t raw[28 * 28 * 3];
        for (int i = 0; i < 28 * 28 * 3; i++) raw[i] = (uint8_t)(i % 256);
        wubu_md3_output_t out;
        CHECK(wubu_md3_infer(raw, 28, 28, 3, "detect objects", &out) == 0, "e2e detect");
        CHECK(out.detections.n_objects > 0 || out.tools != NULL, "detect or tools produced");
        if (out.caption) { free(out.caption); }
        if (out.detections.objects) { free(out.detections.objects); }
        if (out.tools) { free(out.tools); }
    }

    /* MD08: step generation */
    {
        float ctx[10] = { 1, 0.5f, 0.3f, 0.2f, 0.1f, 0.05f, 0.03f, 0.02f, 0.01f, 0.005f };
        float logits[5];
        CHECK(wubu_md3_step(ctx, 10, logits, 5) == 0, "step");
        CHECK(logits[0] != logits[1] || logits[0] != 0, "logits non-trivial");
    }

    /* MD09: box normalization */
    {
        wubu_md3_detect_t det = { 0.1f, 0.2f, 0.5f, 0.6f, 0.9f, "car" };
        int px_min, py_min, px_max, py_max;
        CHECK(wubu_md3_box_normalize(&det, 100, 100, &px_min, &py_min, &px_max, &py_max) == 0, "box norm");
        CHECK(px_min == 10 && py_min == 20, "pixel coords");
        CHECK(px_max == 50 && py_max == 60, "pixel coords max");
    }

    /* MD10: calibration */
    NEAR(wubu_md3_calibrate(0.5f, 2.0f), 0.707f, 1e-3f);
    NEAR(wubu_md3_calibrate(0.8f, 1.0f), 0.8f, 1e-5f);

    if (failures == 0) printf("ALL MOONDAREAM TESTS PASSED\n");
    else printf("%d MOONDAREAM FAILURES\n", failures);
    return failures ? 1 : 0;
}