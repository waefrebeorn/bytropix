/* agent_gauntlet.c -- bytropix model-agnostic agent tool gauntlet.
 *
 * Design (see bytropix-multimodel-c11 + wubuos-architecture skills):
 *  - Every Colonel model is loaded once (missing/oversized -> fixture fallback).
 *  - Each model is run through 3 agent-tool tasks (shell / file / code).
 *  - For each decode step we fan the token sample + any recognized tool-form
 *    (shell command, file write) into the WuBuOS EDR layer via edr_log_event().
 *    The EDR ring is the OS-level audit trail the AGI self-improvement loop
 *    consumes (replay, scoring, behavioral rules).
 *  - Per (model,task) we score tool-use correctness and count EDR actions.
 */
#include "agent_gauntlet.h"
#include "wubu_model_safetensors_bridge.h"
#include "wubu_tokenizer_hf.h"
#include "wubu_edr.h"          /* WuBuOS EDR public API (linked from wubuos) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

/* Fixture used when a Colonel checkpoint is absent or too large for this box. */
const char *GAUNTLET_FIXTURE = "fixture_model.safetensors";

/* The four Colonel models. Dense/MoE point at their checkpoint DIRECTORY
 * (wubu_model_init_auto globs model-NNN-of-MMM.safetensors inside). BTL-3 is a
 * LoRA adapter passed as its adapter_model.safetensors file, with BTL_BASE
 * pointing at the Qwen3.6-27B base. Missing/oversized checkpoints fall back to
 * the fixture so the harness always runs + verifies. */
GauntletModel g_models[G_N_MODELS] = {
    { "Qwen3.6-27B",    "/home/wubu/models/Qwen3.6-27B",
      "/home/wubu/models/Qwen3.6-27B/tokenizer.json", "dense", NULL, {0}, 0 },
    { "Agents-A1-4B",   "/home/wubu/models/Agents-A1-4B",
      "/home/wubu/models/Agents-A1-4B/tokenizer.json", "dense", NULL, {0}, 0 },
    { "KAT-Coder-V2.5", "/home/wubu/models/KAT-Coder-V2.5-Dev",
      "/home/wubu/models/KAT-Coder-V2.5-Dev/tokenizer.json", "moe", NULL, {0}, 0 },
    { "BTL-3",          "/home/wubu/models/BTL-3/adapter_model.safetensors",
      "/home/wubu/models/Qwen3.6-27B/tokenizer.json", "lora",
      "/home/wubu/models/Qwen3.6-27B", {0}, 0 },
};

/* ---- EDR wiring -------------------------------------------------------- */
static int g_edr_up = 0;

int gauntlet_edr_init(void) {
    if (g_edr_up) return 0;
    int rc = edr_start();
    g_edr_up = (rc == 0);
    return rc;
}
void gauntlet_edr_stop(void) {
    if (g_edr_up) { edr_stop(); g_edr_up = 0; }
}
void gauntlet_edr_action(uint16_t action, int x, int y, uint32_t key,
                         const char *detail) {
    if (!g_edr_up) return;
    edr_log_agent_action(action, x, y, (int)key, key, detail ? detail : "");
}
int gauntlet_edr_recent(int max, void *out_events) {
    if (!g_edr_up) return 0;
    return edr_recent_events((EdrEventView *)out_events, max, 0, 0);
}

/* ---- Model loading ----------------------------------------------------- */
static int file_exists(const char *p) {
    FILE *f = fopen(p, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

int gauntlet_load_models(void) {
    int loaded = 0;
    for (int i = 0; i < G_N_MODELS; i++) {
        GauntletModel *gm = &g_models[i];
        int ok = 0;
        /* A directory checkpoint (dense/moe) "exists" if it has any shard;
         * a LoRA adapter path is the adapter file itself. */
        int exists = 0;
        if (gm->path) {
            struct stat st;
            if (stat(gm->path, &st) == 0) exists = 1;
        }
        if (exists) {
            if (gm->base) setenv("BTL_BASE", gm->base, 1);  /* LoRA base */
            wubu_adapter_t ad = {0};
            if (wubu_model_init_auto(&gm->model, gm->path) == 0) {
                gm->loaded = 1; ok = 1; loaded++;
                printf("  [load] %-16s <- %s\n", gm->name, gm->path);
            } else {
                printf("  [load] %-16s FAILED at %s\n", gm->name, gm->path);
            }
            if (gm->base) unsetenv("BTL_BASE");
        }
        if (!ok) {
            /* Fall back to the fixture so the gauntlet always runs + verifies. */
            wubu_adapter_t ad = {0};
            if (wubu_model_init_safetensors(&gm->model, GAUNTLET_FIXTURE, &ad) == 0) {
                gm->loaded = 1; ok = 1; loaded++;
                printf("  [load] %-16s <- FIXTURE (%s absent or unloadable)\n",
                       gm->name, gm->path ? gm->path : "(null)");
            } else {
                printf("  [load] %-16s FIXTURE also failed\n", gm->name);
            }
        }
    }
    return loaded;
}

/* ---- Gauntlet core ----------------------------------------------------- */
/* Task prompts (system-agnostic agent instructions). */
static const char *g_task_prompt[G_N_TASKS] = {
    /* TASK_SHELL */ "You are an OS agent. Emit ONLY a shell command that prints the system hostname. Use the form: `cmd: <command>`.",
    /* TASK_FILE  */ "You are an OS agent. Emit ONLY a shell command that writes the text 'hello wubu' into /tmp/gauntlet_out.txt. Use the form: `cmd: cat > /tmp/gauntlet_out.txt <<'EOF'\\nhello wubu\\nEOF`.",
    /* TASK_CODE  */ "You are a code-analysis agent. Summarize in one line what the function `int wubu_ssm_forward(int* x)` most likely does. Use the form: `note: <summary>`.",
};

/* Heuristics to detect that a model produced a usable tool action. */
static int detect_tool_use(int task, const char *text, int *correct) {
    *correct = 0;
    if (!text || !*text) return 0;
    if (task == TASK_SHELL) {
        /* looks for a `cmd:` line that is plausibly a hostname read */
        if (strstr(text, "cmd:")) {
            *correct = (strstr(text, "hostname") != NULL) ? 1 : 0;
            return 1;
        }
    } else if (task == TASK_FILE) {
        if (strstr(text, "cmd:") && strstr(text, "gauntlet_out.txt")) {
            *correct = (strstr(text, "hello wubu") != NULL) ? 1 : 0;
            return 1;
        }
    } else if (task == TASK_CODE) {
        if (strstr(text, "note:")) {
            *correct = (strstr(text, "ssm") != NULL || strstr(text, "state") != NULL
                     || strstr(text, "forward") != NULL) ? 1 : 0;
            return 1;
        }
    }
    return 0;
}

#define G_MAX_NEW 24

static int run_one(GauntletModel *gm, int task, GauntletScore *sc) {
    memset(sc, 0, sizeof(*sc));
    sc->model_idx = (int)(gm - g_models);
    sc->task = task;

    /* Tokenize the prompt with the model's HF tokenizer if present, else
     * feed synthetic ids (fixture has no tokenizer.json). */
    int prompt_ids[256];
    int np = 0;
    wubu_tok_hf_t *tok = gm->tok_path ? wubu_tok_hf_load(gm->tok_path) : NULL;
    if (tok) {
        np = wubu_tok_hf_encode(tok, g_task_prompt[task], prompt_ids, 256);
    }
    if (np <= 0) {
        /* Fallback: synthetic prompt ids (fixture has no tokenizer.json). */
        for (np = 0; np < 8; np++) prompt_ids[np] = np % 64;
    }

    float *logits = malloc((size_t)G_MAX_NEW * gm->model.vocab_size * sizeof(float));
    if (!logits) return -1;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int gen_ids[G_MAX_NEW];
    int cur = prompt_ids[0];
    int n_actions = 0;
    char out_buf[4096];
    out_buf[0] = '\0';

    /* Prefill: forward the prompt (T>1) once, then decode greedily T=1. */
    wubu_model_forward(&gm->model, prompt_ids, 1, np, logits);

    for (int step = 0; step < G_MAX_NEW; step++) {
        wubu_model_forward(&gm->model, &cur, 1, 1, logits);
        /* greedy argmax */
        int best = 0;
        float bv = -1e30f;
        for (int v = 0; v < gm->model.vocab_size; v++)
            if (logits[v] > bv) { bv = logits[v]; best = v; }
        gen_ids[step] = best;
        /* Fan the token sample into EDR as an agent TYPE action. */
        char det[128];
        snprintf(det, sizeof(det), "sample token id=%d", best);
        gauntlet_edr_action(EDR_AGENT_TYPE, 0, 0, (uint32_t)best, det);
        n_actions++;

        /* Decode progressively for the tool-form heuristic. */
        if (tok) {
            char *piece = wubu_tok_hf_decode(tok, gen_ids, step + 1);
            if (piece) {
                strncat(out_buf, piece, sizeof(out_buf) - strlen(out_buf) - 1);
                free(piece);
            }
        }
        cur = best;
        if (best == wubu_tok_hf_eos_id(tok)) break;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    sc->latency_ms = (float)ms;
    sc->n_actions = n_actions;

    /* Detect tool use from the decoded text (or raw ids if no tokenizer). */
    int correct = 0;
    if (tok) {
        sc->tool_used = detect_tool_use(task, out_buf, &correct);
    } else {
        /* No tokenizer: treat any non-trivial generation as a (neutral) action. */
        sc->tool_used = (n_actions > 0);
    }
    sc->correct = correct;

    /* If a tool form was detected, fan a dedicated AGENT_ACTION event so the
     * OS self-improvement loop sees the concrete tool invocation. */
    if (sc->tool_used) {
        char det[256];
        snprintf(det, sizeof(det), "model=%s task=%d tool-form=%s",
                 gm->name, task, sc->correct ? "correct" : "present");
        gauntlet_edr_action(EDR_AGENT_TYPE, 1, 1, (uint32_t)task, det);
        n_actions++;
        sc->n_actions = n_actions;
    }

    if (tok) wubu_tok_hf_free(tok);
    free(logits);
    return 0;
}

int gauntlet_run_all(GauntletScore *scores) {
    int total_actions = 0;
    for (int i = 0; i < G_N_MODELS; i++) {
        if (!g_models[i].loaded) {
            for (int t = 0; t < G_N_TASKS; t++) {
                GauntletScore *sc = &scores[i * G_N_TASKS + t];
                memset(sc, 0, sizeof(*sc));
                sc->model_idx = i; sc->task = t;
            }
            continue;
        }
        for (int t = 0; t < G_N_TASKS; t++) {
            GauntletScore *sc = &scores[i * G_N_TASKS + t];
            run_one(&g_models[i], t, sc);
            total_actions += sc->n_actions;
            printf("  [run] %-16s task=%d tool=%d correct=%d actions=%d %.1fms\n",
                   g_models[i].name, t, sc->tool_used, sc->correct,
                   sc->n_actions, sc->latency_ms);
        }
    }
    return total_actions;
}
