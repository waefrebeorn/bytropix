/* wubu_modelcard.c -- BRAIN #128 math model card (AGI disclosure).
 * See wubu_modelcard.h. */
#include "wubu_modelcard.h"
#include <string.h>
#include <stdio.h>

int wubu_mcard_init(wubu_modelcard_t *mc, const char *name, const char *version,
                    int n_params_m, const char *intended_use) {
    if (!mc || !name || !version || !intended_use) return -1;
    memset(mc, 0, sizeof *mc);
    if (strlen(name) >= WUBU_MCARD_NAME) return -1;
    if (strlen(version) >= 16) return -1;
    if (strlen(intended_use) >= WUBU_MCARD_TEXT) return -1;
    strncpy(mc->name, name, WUBU_MCARD_NAME - 1);
    strncpy(mc->version, version, 15);
    strncpy(mc->intended_use, intended_use, WUBU_MCARD_TEXT - 1);
    mc->n_params_m = n_params_m;
    mc->n_items = 0;
    return 0;
}

int wubu_mcard_add(wubu_modelcard_t *mc, wubu_mcard_kind_t kind, const char *text) {
    if (!mc || !text || mc->n_items >= WUBU_MCARD_MAX_ITEMS) return -1;
    if (kind < WUBU_MCARD_CAPABILITY || kind > WUBU_MCARD_EVALUATION) return -1;
    if (strlen(text) >= WUBU_MCARD_TEXT) return -1;
    strncpy(mc->items[mc->n_items], text, WUBU_MCARD_TEXT - 1);
    mc->kinds[mc->n_items] = kind;
    mc->n_items++;
    return 0;
}

int wubu_mcard_count(const wubu_modelcard_t *mc, wubu_mcard_kind_t kind) {
    if (!mc) return 0;
    int c = 0;
    for (int i = 0; i < mc->n_items; i++) if (mc->kinds[i] == kind) c++;
    return c;
}

int wubu_mcard_render(const wubu_modelcard_t *mc, char *out, int cap) {
    if (!mc || !out || cap <= 0) return -1;
    int len = 0;
    len += snprintf(out + len, cap - len, "# Model Card: %s v%s\n", mc->name, mc->version);
    len += snprintf(out + len, cap - len, "Parameters: %dM\n", mc->n_params_m);
    len += snprintf(out + len, cap - len, "Intended use: %s\n\n", mc->intended_use);
    const char *titles[3] = { "Capabilities", "Limitations", "Evaluation" };
    for (int k = 0; k < 3; k++) {
        len += snprintf(out + len, cap - len, "## %s\n", titles[k]);
        int any = 0;
        for (int i = 0; i < mc->n_items; i++) {
            if (mc->kinds[i] == k) {
                len += snprintf(out + len, cap - len, "- %s\n", mc->items[i]);
                any = 1;
            }
        }
        if (!any) len += snprintf(out + len, cap - len, "- (none disclosed)\n");
        len += snprintf(out + len, cap - len, "\n");
    }
    return len;
}

int wubu_mcard_disclosable(const wubu_modelcard_t *mc) {
    if (!mc) return 0;
    return (wubu_mcard_count(mc, WUBU_MCARD_CAPABILITY) >= 1 &&
            wubu_mcard_count(mc, WUBU_MCARD_LIMITATION) >= 1 &&
            wubu_mcard_count(mc, WUBU_MCARD_EVALUATION) >= 1) ? 1 : 0;
}
