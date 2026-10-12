/*
 * wubu_tool_cap.c -- capability-gated tool dispatch (AX04 + AD01).
 *
 * wubu_tool_call() dispatches a registered tool with NO capability check:
 * any caller can invoke any registered tool. For an agentic OS that is the
 * exact hole the Styx private-namespace-as-capability model closes. This
 * module wraps every tool call with an AD01 9P capability check: a tool is
 * bound to a capability path (its subtree), and a call is refused unless
 * the caller's subtree owns that path (wubu_9p_cap_allowed).
 *
 * The rule: the agent's namespace attach set IS its capability set. A tool
 * bound to /n/companion/tools runs only for a caller whose subtree is
 * /n/companion (or an ancestor). An unrelated agent is denied before the
 * handler ever runs. This is the security spine the companion (Track B)
 * and foreign-ABI translation (Track D) both reuse.
 *
 * C11, minimal includes. Builds on wubu_tooluse (dispatch) + wubu_agentic_os
 * (AD01).
 */

#include "wubu_tool_cap.h"
#include "wubu_tooluse.h"
#include "wubu_agentic_os.h"   /* AD01: wubu_9p_cap_allowed */
#include <string.h>
#include <stdio.h>

/* Bind a registered tool to a capability path. Returns 0 on success, -1 if
 * the registry is full or the tool isn't found. */
int wubu_tool_cap_bind(wubu_tool_cap_t *caps, const char *tool_name,
                       const char *cap_path) {
    if (!caps || !tool_name || !cap_path) return -1;
    if (caps->n >= WUBU_TOOL_CAP_MAX) return -1;
    wubu_tool_cap_entry_t *e = &caps->entries[caps->n];
    snprintf(e->tool_name, sizeof(e->tool_name), "%s", tool_name);
    snprintf(e->cap_path, sizeof(e->cap_path), "%s", cap_path);
    caps->n++;
    return 0;
}

/* Look up the capability path bound to a tool (NULL if unbound). */
static const char *cap_for(const wubu_tool_cap_t *caps, const char *tool_name) {
    if (!caps) return NULL;
    for (int i = 0; i < caps->n; i++)
        if (strcmp(caps->entries[i].tool_name, tool_name) == 0)
            return caps->entries[i].cap_path;
    return NULL;
}

/* Capability-gated dispatch. Returns:
 *   0 / >0  = the handler's result (call allowed)
 *   -1      = bad args / unknown tool / no handler (same as wubu_tool_call)
 *   -2      = DENIED: the caller's subtree does not own the tool's cap path
 * The denial is fail-closed and the handler is never invoked. */
int wubu_tool_cap_call(const wubu_tool_cap_t *caps, wubu_tool_registry_t *reg,
                       const char *caller_subtree, const char *name,
                       const char *args_json, char *result_buf, int buf_size) {
    if (!reg || !name || !args_json) return -1;
    const char *cap_path = cap_for(caps, name);
    /* A tool with NO bound capability is deny-by-default: refuse it. This
     * is the safe direction -- a tool must be explicitly granted a subtree
     * before an agent may call it. */
    if (!cap_path) return -2;
    if (!wubu_9p_cap_allowed(caller_subtree, cap_path)) return -2;
    return wubu_tool_call(reg, name, args_json, result_buf, buf_size);
}
