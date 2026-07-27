/* agent_gauntlet.h -- bytropix model-agnostic agent tool gauntlet
 *
 * Runs every loaded Colonel model through a battery of agent-tool tasks and
 * fans every agent action (tool invocation, token sample, completion) into the
 * WuBuOS EDR layer (src/runtime/edr) for the OS-level AGI self-improvement /
 * observability loop.
 *
 * C11, self-contained. The EDR sources are linked directly (no daemon): the
 * gauntlet calls edr_start() + edr_log_event() from wubu_edr.h.
 */
#ifndef AGENT_GAUNTLET_H
#define AGENT_GAUNTLET_H

#include <stdint.h>
#include <stddef.h>
#include "wubu_model.h"   /* bytropix engine: wubu_model_t, wubu_model_init_auto, ... */
#include "wubu_edr.h"     /* WuBuOS EDR public API (EDR_EV_*, EdrEventView, edr_log_*) */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Model registry ---------------------------------------------------- */
#define G_N_MODELS 4

typedef struct {
    const char *name;       /* Colonel codename */
    const char *path;       /* on-disk checkpoint: dir (dense/moe) or adapter file (lora) */
    const char *tok_path;   /* tokenizer.json (NULL = none / synthetic fallback) */
    const char *kind;       /* "dense" | "moe" | "lora" | "fixture" */
    const char *base;       /* BTL_BASE for lora, else NULL */
    wubu_model_t model;     /* zeroed at start; filled by gauntlet_load_models */
    int          loaded;    /* 1 if model->loaded */
} GauntletModel;

/* The four Colonel models. Paths resolve at runtime; missing ones fall back
 * to the fixture (GAUNTLET_FIXTURE) so the harness always runs + verifies. */
extern GauntletModel g_models[G_N_MODELS];
extern const char   *GAUNTLET_FIXTURE;

/* Load every model; missing/oversized ones fall back to the fixture.
 * Prints a per-model load report. Returns number of models actually loaded. */
int gauntlet_load_models(void);

/* ---- Gauntlet tasks ---------------------------------------------------- */
#define G_N_TASKS 3
typedef enum {
    TASK_SHELL = 0,   /* agent must emit a shell command to read /etc/hostname */
    TASK_FILE  = 1,   /* agent must emit a file write (cat <<'EOF' > out.txt)  */
    TASK_CODE  = 2,   /* agent must emit a code-analysis diff/summary          */
} GauntletTask;

/* Per-(model,task) score: how well the model used the tool correctly. */
typedef struct {
    int   model_idx;
    int   task;
    int   tool_used;     /* 1 if the model emitted the expected tool form */
    int   correct;       /* 1 if the emitted action would achieve the goal */
    int   n_actions;     /* number of EDR agent actions fanned out */
    float latency_ms;    /* wall time for the forward+decode of this task */
} GauntletScore;

/* Run all (model,task) combos. scores[] must hold G_N_MODELS*G_N_TASKS slots.
 * Returns total agent actions fanned into EDR. */
int gauntlet_run_all(GauntletScore *scores);

/* ---- EDR fan-out (WuBuOS AGI self-improvement layer) ------------------- */
/* Initialize the EDR engine (lock-free ring + behavioral modules). Safe to
 * call once at startup. Returns 0 on success. */
int  gauntlet_edr_init(void);
void gauntlet_edr_stop(void);

/* Log one agent action to EDR (mirrors the OS-side UI-automation disclosure
 * model). action = EDR_AGENT_* sub-type; detail is a human summary. */
void gauntlet_edr_action(uint16_t action, int x, int y, uint32_t key,
                         const char *detail);

/* Snapshot the most recent EDR events (for the self-improvement audit loop). */
int  gauntlet_edr_recent(int max, void *out_events);

#ifdef __cplusplus
}
#endif
#endif /* AGENT_GAUNTLET_H */
