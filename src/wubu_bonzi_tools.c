/* wubu_bonzi_tools.c -- UX #240 companion capability-gated tools.
 * See wubu_bonzi_tools.h. */
#include "wubu_bonzi_tools.h"
#include <string.h>

int wubu_bt_init(wubu_bonzi_tools_t *bt) {
    if (!bt) return -1;
    memset(bt->tools, 0, sizeof bt->tools);
    memset(bt->caps, 0, sizeof bt->caps);
    bt->n_tools = 0;
    bt->n_caps = 0;
    bt->n_allowed = 0;
    bt->n_denied = 0;
    return 0;
}

int wubu_bt_grant(wubu_bonzi_tools_t *bt, const char *cap) {
    if (!bt || !cap || bt->n_caps >= WUBU_BT_MAX_CAPS) return -1;
    if (strlen(cap) >= WUBU_BT_NAMELEN) return -1;
    for (int i = 0; i < bt->n_caps; i++)
        if (strcmp(bt->caps[i], cap) == 0) return 0;   /* already granted */
    strncpy(bt->caps[bt->n_caps++], cap, WUBU_BT_NAMELEN - 1);
    return 0;
}

int wubu_bt_register(wubu_bonzi_tools_t *bt, const char *tool, const char *cap) {
    if (!bt || !tool || !cap || bt->n_tools >= WUBU_BT_MAX_TOOLS) return -1;
    if (strlen(tool) >= WUBU_BT_NAMELEN || strlen(cap) >= WUBU_BT_NAMELEN) return -1;
    strncpy(bt->tools[bt->n_tools].name, tool, WUBU_BT_NAMELEN - 1);
    strncpy(bt->tools[bt->n_tools].cap, cap, WUBU_BT_NAMELEN - 1);
    bt->n_tools++;
    return 0;
}

static const wubu_bt_tool_t *find_tool(const wubu_bonzi_tools_t *bt, const char *tool) {
    for (int i = 0; i < bt->n_tools; i++)
        if (strcmp(bt->tools[i].name, tool) == 0) return &bt->tools[i];
    return NULL;
}

static int has_cap(const wubu_bonzi_tools_t *bt, const char *cap) {
    for (int i = 0; i < bt->n_caps; i++)
        if (strcmp(bt->caps[i], cap) == 0) return 1;
    return 0;
}

int wubu_bt_may_invoke(const wubu_bonzi_tools_t *bt, const char *tool) {
    if (!bt || !tool) return 0;   /* fail-closed */
    const wubu_bt_tool_t *t = find_tool(bt, tool);
    if (!t) return 0;   /* unknown tool: fail-closed */
    return has_cap(bt, t->cap) ? 1 : 0;
}

int wubu_bt_invoke(wubu_bonzi_tools_t *bt, const char *tool) {
    if (!bt || !tool) return 0;
    if (wubu_bt_may_invoke(bt, tool)) { bt->n_allowed++; return 1; }
    bt->n_denied++;
    return 0;
}

int wubu_bt_allowed(const wubu_bonzi_tools_t *bt) { return bt ? bt->n_allowed : 0; }
int wubu_bt_denied(const wubu_bonzi_tools_t *bt) { return bt ? bt->n_denied : 0; }
