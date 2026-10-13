/* wubu_hypr_perm.c -- OMARCHY #281 Hyprland permission management protocol.
 * See wubu_hypr_perm.h. Real, deterministic fail-closed permission gate. */

#include "wubu_hypr_perm.h"
#include <string.h>

static int match(const wubu_hp_prompt_t *p, wubu_hp_op_t op,
                 const char *rq, const char *res) {
    if (!p->always) return 0;
    return p->op == op &&
           strcmp(p->requester, rq) == 0 &&
           strcmp(p->resource, res) == 0;
}

int wubu_hp_init(wubu_hypr_perm_t *hp) {
    if (!hp) return -1;
    memset(hp, 0, sizeof(*hp));
    hp->next_id = 1;
    return 0;
}

int wubu_hp_has_grant(const wubu_hypr_perm_t *hp, wubu_hp_op_t op,
                      const char *requester, const char *resource) {
    if (!hp || !requester || !resource) return 0;
    for (int i = 0; i < hp->n_prompts; i++)
        if (match(&hp->prompts[i], op, requester, resource)) return 1;
    return 0;
}

static wubu_hp_prompt_t *find_active(wubu_hypr_perm_t *hp, wubu_hp_op_t op,
                                     const char *rq, const char *res) {
    for (int i = 0; i < hp->n_prompts; i++) {
        wubu_hp_prompt_t *p = &hp->prompts[i];
        if (p->active && p->op == op &&
            strcmp(p->requester, rq) == 0 &&
            strcmp(p->resource, res) == 0)
            return p;
    }
    return NULL;
}

int wubu_hp_request(wubu_hypr_perm_t *hp, wubu_hp_op_t op,
                    const char *requester, const char *resource) {
    if (!hp || !requester || !resource) return -1;
    if (op < 0 || op >= WUBU_HP_OP_COUNT) return -1;

    /* persisted always-grant admits immediately */
    if (wubu_hp_has_grant(hp, op, requester, resource)) {
        hp->n_admitted++;
        return 1;
    }

    /* is there an outstanding prompt for this exact request? */
    wubu_hp_prompt_t *p = find_active(hp, op, requester, resource);
    if (p) {
        /* re-request while pending: still denied (no decision yet) */
        hp->n_denied++;
        return 0;
    }

    /* raise a new prompt (fail-closed until answered YES) */
    if (hp->n_prompts >= WUBU_HP_MAX_PROMPTS) {
        hp->n_denied++;   /* fail-closed on prompt-table exhaustion */
        return 0;
    }
    wubu_hp_prompt_t *np = &hp->prompts[hp->n_prompts];
    memset(np, 0, sizeof(*np));
    np->op = op;
    np->id = hp->next_id++;
    np->active = 1;
    strncpy(np->requester, requester, WUBU_HP_NAME_MAX - 1);
    strncpy(np->resource, resource, WUBU_HP_NAME_MAX - 1);
    hp->n_prompts++;
    hp->n_asked++;
    hp->n_denied++;   /* default-deny: pending request is denied */
    return 0;
}

int wubu_hp_answer(wubu_hypr_perm_t *hp, int prompt_id, wubu_hp_decision_t decision) {
    if (!hp || prompt_id <= 0) return -1;
    for (int i = 0; i < hp->n_prompts; i++) {
        wubu_hp_prompt_t *p = &hp->prompts[i];
        if (p->id == prompt_id && p->active) {
            p->active = 0;
            p->decision = decision;
            if (decision == WUBU_HP_DECISION_ALLOW) {
                p->always = 0;
                hp->n_admitted++;
            } else if (decision == WUBU_HP_DECISION_DENY) {
                hp->n_denied++;
            } else if (decision == WUBU_HP_DECISION_ALLOW_ALWAYS) {
                p->always = 1;
                p->decision = WUBU_HP_DECISION_ALLOW;
                hp->n_admitted++;
            } else {
                return -1;   /* invalid decision */
            }
            return 0;
        }
    }
    return -1;   /* no active prompt with that id */
}

int wubu_hp_revoke(wubu_hypr_perm_t *hp, const char *requester) {
    if (!hp || !requester) return -1;
    int n = 0;
    for (int i = 0; i < hp->n_prompts; i++) {
        if (hp->prompts[i].always &&
            strcmp(hp->prompts[i].requester, requester) == 0) {
            hp->prompts[i].always = 0;
            n++;
        }
    }
    return n;
}

int wubu_hp_pending(const wubu_hypr_perm_t *hp) {
    if (!hp) return -1;
    int n = 0;
    for (int i = 0; i < hp->n_prompts; i++)
        if (hp->prompts[i].active) n++;
    return n;
}
