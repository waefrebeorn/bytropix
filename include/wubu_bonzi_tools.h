#ifndef WUBU_BONZI_TOOLS_H
#define WUBU_BONZI_TOOLS_H
#include <stddef.h>

/* wubu_bonzi_tools.h -- UX #240 companion capability-gated tools.
 *
 * The companion can only invoke tools it holds a capability for (default-deny).
 * Each tool has a name and a required capability; the companion's capability
 * set determines which tools it may call. Invoking an ungranted tool is
 * refused (fail-closed). This ties the companion's agency to the object-
 * capability model. Deterministic, fixed-size, real work (no stubs). */

#define WUBU_BT_MAX_TOOLS 32
#define WUBU_BT_MAX_CAPS 16
#define WUBU_BT_NAMELEN 40

typedef struct {
    char name[WUBU_BT_NAMELEN];
    char cap[WUBU_BT_NAMELEN];   /* required capability */
} wubu_bt_tool_t;

typedef struct {
    wubu_bt_tool_t tools[WUBU_BT_MAX_TOOLS];
    int n_tools;
    char caps[WUBU_BT_MAX_CAPS][WUBU_BT_NAMELEN];
    int n_caps;
    /* telemetry */
    int n_allowed;
    int n_denied;
} wubu_bonzi_tools_t;

/* Initialise the companion's tool registry (no caps, no tools). */
int wubu_bt_init(wubu_bonzi_tools_t *bt);

/* Grant the companion a capability. Returns 0/-1. */
int wubu_bt_grant(wubu_bonzi_tools_t *bt, const char *cap);

/* Register a tool requiring a capability. Returns 0/-1. */
int wubu_bt_register(wubu_bonzi_tools_t *bt, const char *tool, const char *cap);

/* May the companion invoke this tool? (has the required cap). 1/0. */
int wubu_bt_may_invoke(const wubu_bonzi_tools_t *bt, const char *tool);

/* Invoke a tool: returns 1 if allowed (and counts it), 0 if denied
 * (fail-closed). */
int wubu_bt_invoke(wubu_bonzi_tools_t *bt, const char *tool);

/* Telemetry. */
int wubu_bt_allowed(const wubu_bonzi_tools_t *bt);
int wubu_bt_denied(const wubu_bonzi_tools_t *bt);

#endif
