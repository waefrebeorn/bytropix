/*
 * wubu_tooluse.c -- MCP-compatible tool schema + dispatch (AX04). C11.
 *
 * Convergence (MCP/function-calling 7-hop):
 *   - AX04: tool schema registry -- name+description+JSON Schema input,
 *     parallel dispatch, result aggregation.
 */
#include "wubu_tooluse.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WUBU_TOOL_MAX_TOOLS 64
#define WUBU_TOOL_MAX_NAME 64
#define WUBU_TOOL_MAX_SCHEMA 512

int wubu_tool_register(wubu_tool_registry_t *reg, const char *name,
                                   const char *description,
                                   const char *input_schema) {
    if (!reg || !name || !input_schema) return -1;
    if (reg->n_tools >= WUBU_TOOL_MAX_TOOLS) return -1;
    if (strlen(name) >= WUBU_TOOL_MAX_NAME) return -1;
    if (strlen(input_schema) >= WUBU_TOOL_MAX_SCHEMA) return -1;
    wubu_tool_t *t = &reg->tools[reg->n_tools++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->description = description;
    t->input_schema = input_schema;
    t->handler = NULL;
    t->ctx = NULL;
    return 0;
}

int wubu_tool_registry_init(wubu_tool_registry_t *reg) {
    if (!reg) return -1;
    reg->n_tools = 0;
    return 0;
}

int wubu_tool_set_handler(wubu_tool_registry_t *reg, const char *name,
                              wubu_tool_handler_t handler, void *ctx) {
    if (!reg || !name || !handler) return -1;
    for (int i = 0; i < reg->n_tools; i++) {
        if (strcmp(reg->tools[i].name, name) == 0) {
            reg->tools[i].handler = handler;
            reg->tools[i].ctx = ctx;
            return 0;
        }
    }
    return -1;  /* tool not found */
}

/* ---- Dispatch ---- */
static wubu_tool_t *find_tool(wubu_tool_registry_t *reg, const char *name) {
    for (int i = 0; i < reg->n_tools; i++)
        if (strcmp(reg->tools[i].name, name) == 0) return &reg->tools[i];
    return NULL;
}

int wubu_tool_call(wubu_tool_registry_t *reg, const char *name,
                       const char *args_json, char *result_buf, int buf_size) {
    if (!reg || !name || !args_json) return -1;
    wubu_tool_t *t = find_tool(reg, name);
    if (!t || !t->handler) return -1;
    return t->handler(t->ctx, args_json, result_buf, buf_size);
}

int wubu_tool_parallel_call(wubu_tool_registry_t *reg, const char **names,
                                const char **args, int n,
                                char *results, int results_size) {
    if (!reg || !names || !args || n <= 0) return -1;
    int total = 0;
    for (int i = 0; i < n; i++) {
        int off = total;
        int rc = wubu_tool_call(reg, names[i], args[i],
                                results + off, results_size - off);
        if (rc < 0) return rc;
        total += rc;
        if (total >= results_size) break;
    }
    return total;
}