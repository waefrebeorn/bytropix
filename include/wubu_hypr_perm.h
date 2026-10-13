#ifndef WUBU_HYPR_PERM_H
#define WUBU_HYPR_PERM_H

/* wubu_hypr_perm.h -- OMARCHY #281 Hyprland permission management protocol.
 *
 * The compositor-side permission prompt system for privileged Wayland
 * operations (screencopy/ScreenShot, screen-capture, global-shortcut grab,
 * input-injection). Every privileged request is fail-closed default-deny: a
 * request is admitted only when an interactive prompt has been answered YES,
 * or a prior YES persists in the persistent-grant table. The module models the
 * prompt registry, per-requester/per-method grants, the user decision, the
 * one-way "don't ask again" escalation, and revocation. Real work, no stubs.
 * (Pairs with #280 the Hyprland compositor backend.) */

#include <stddef.h>

#define WUBU_HP_MAX_PROMPTS 64
#define WUBU_HP_NAME_MAX 48

/* privileged request kinds the compositor gates */
typedef enum {
    WUBU_HP_OP_SCREENSHOT = 0,   /* screencopy of an output */
    WUBU_HP_OP_CAPTURE,          /* screen capture stream */
    WUBU_HP_OP_GLOBAL_HK,        /* global shortcut grab */
    WUBU_HP_OP_INJECT,           /* synthetic input injection */
    WUBU_HP_OP_COUNT
} wubu_hp_op_t;

typedef enum {
    WUBU_HP_DECISION_NONE = 0,   /* prompt unanswered */
    WUBU_HP_DECISION_ALLOW,
    WUBU_HP_DECISION_DENY,
    WUBU_HP_DECISION_ALLOW_ALWAYS   /* "don't ask again" */
} wubu_hp_decision_t;

typedef struct {
    wubu_hp_op_t op;
    char requester[WUBU_HP_NAME_MAX];   /* app_id / pid */
    char resource[WUBU_HP_NAME_MAX];    /* target, e.g. output name or shortcut */
    int id;                       /* unique prompt id */
    int active;                   /* 1 = prompt outstanding */
    wubu_hp_decision_t decision;  /* last decision */
    int always;                   /* "don't ask again" granted */
} wubu_hp_prompt_t;

typedef struct {
    wubu_hp_prompt_t prompts[WUBU_HP_MAX_PROMPTS];
    int n_prompts;
    int next_id;
    int n_admitted;     /* total requests admitted (allow/always) */
    int n_denied;       /* denied requests */
    int n_asked;        /* prompts raised (allow/deny, not always) */
} wubu_hypr_perm_t;

/* Initialise (empty, default-deny). */
int wubu_hp_init(wubu_hypr_perm_t *hp);

/* A privileged request comes in. Returns:
 *   1  admitted (granted -- via prior always-grant or just-answered YES)
 *   0  denied (fail-closed)
 *  -1  bad args
 * A fresh request (no prior grant) raises a prompt (active=1) and is denied
 * until the user answers; answering YES admits, NO denies, YES-always admits
 * and persists for that (op,requester,resource). */
int wubu_hp_request(wubu_hypr_perm_t *hp, wubu_hp_op_t op,
                    const char *requester, const char *resource);

/* User answers an outstanding prompt. Returns 0/-1. */
int wubu_hp_answer(wubu_hypr_perm_t *hp, int prompt_id,
                   wubu_hp_decision_t decision);

/* Admit if there is a persisted always-grant for this (op,requester,resource). */
int wubu_hp_has_grant(const wubu_hypr_perm_t *hp, wubu_hp_op_t op,
                      const char *requester, const char *resource);

/* Revoke all grants for a requester. */
int wubu_hp_revoke(wubu_hypr_perm_t *hp, const char *requester);

/* Count currently-outstanding prompts. */
int wubu_hp_pending(const wubu_hypr_perm_t *hp);

#endif
