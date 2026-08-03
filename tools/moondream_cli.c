/*
 * moondream_cli.c -- MoonDream 3 operational runner (the agentic vision core).
 *
 * Loads a real image (PPM P6 binary / P5 gray), runs the FULL MoonDream
 * pipeline end-to-end (preprocess -> encode -> MoE -> detect/caption/
 * toolcall), and prints the structured result. This is the operational
 * front-end for wubu_moondream -- the same pipeline the kernel's ring-0
 * Colonel will call with a framebuffer capture.
 *
 * Usage: moondream_cli <image.ppm> [prompt]
 *   prompt defaults to "describe" (caption path); "detect <object>"
 *   runs the detection head.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_moondream.h"

static uint8_t *load_ppm(const char *path, int *w, int *h, int *c)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[3] = { 0 };
    if (fscanf(f, "%2s", magic) != 1) { fclose(f); return NULL; }
    if (strcmp(magic, "P6") != 0 && strcmp(magic, "P5") != 0) {
        fprintf(stderr, "moondream_cli: only P5 (gray) / P6 (rgb) PPM supported\n");
        fclose(f); return NULL;
    }
    int maxval = 0;
    if (fscanf(f, "%d %d %d", w, h, &maxval) != 3) { fclose(f); return NULL; }
    if (*w <= 0 || *h <= 0 || maxval != 255) { fclose(f); return NULL; }
    int chans = (magic[1] == '6') ? 3 : 1;
    *c = chans;
    /* skip a single whitespace before the raster */
    int ch = fgetc(f);
    if (ch == EOF) { fclose(f); return NULL; }
    uint8_t *px = (uint8_t *)malloc(sizeof(uint8_t) * (size_t)(*w) * (*h) * chans);
    if (!px) { fclose(f); return NULL; }
    size_t got = fread(px, 1, (size_t)(*w) * (*h) * chans, f);
    fclose(f);
    if (got != (size_t)(*w) * (*h) * chans) { free(px); return NULL; }
    return px;
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : NULL;
    const char *prompt = (argc > 2) ? argv[2] : "describe";
    if (!path) {
        fprintf(stderr, "usage: %s <image.ppm> [describe|detect <obj>]\n", argv[0]);
        return 2;
    }

    int w, h, c;
    uint8_t *raw = load_ppm(path, &w, &h, &c);
    if (!raw) {
        fprintf(stderr, "moondream_cli: cannot load %s\n", path);
        return 1;
    }
    printf("moondream: loaded %s (%dx%d, %d chan)\n", path, w, h, c);

    wubu_md3_output_t out;
    memset(&out, 0, sizeof(out));
    int rc = wubu_md3_infer(raw, w, h, c, prompt, &out);
    free(raw);
    if (rc != 0) {
        fprintf(stderr, "moondream_cli: infer failed (rc=%d)\n", rc);
        return 1;
    }

    /* the structured result */
    if (out.caption) {
        printf("moondream: caption: %s\n", out.caption);
    }
    if (out.detections.n_objects > 0) {
        printf("moondream: detected %d object(s):\n", out.detections.n_objects);
        for (int i = 0; i < out.detections.n_objects; i++) {
            const wubu_md3_detect_t *d = &out.detections.objects[i];
            printf("  [%d] %s conf=%.2f box=(%.2f,%.2f)-(%.2f,%.2f)\n",
                   i, d->label, d->confidence,
                   d->x_min, d->y_min, d->x_max, d->y_max);
        }
    }
    for (int i = 0; i < out.n_tools; i++) {
        printf("moondream: toolcall: %s %s\n",
               out.tools[i].name, out.tools[i].arguments);
    }
    printf("moondream: done\n");

    if (out.caption) free(out.caption);
    if (out.detections.objects) free(out.detections.objects);
    if (out.tools) free(out.tools);
    return 0;
}
