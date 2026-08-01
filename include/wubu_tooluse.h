/*
 * wubu_tooluse.h -- MCP-compatible tool schema + dispatch (AX04).
 */
#ifndef WUBU_TOOLUSE_H
#define WUBU_TOOLUSE_H

#define WUBU_TOOL_MAX_TOOLS 64
#define WUBU_TOOL_MAX_NAME 64
#define WUBU_TOOL_MAX_SCHEMA 512

typedef int (*wubu_tool_handler_t)(void *ctx, const char *args_json,
                                            char *result_buf, int buf_size);

typedef struct {
    char name[WUBU_TOOL_MAX_NAME];
    const char *description;
    const char *input_schema;
    wubu_tool_handler_t handler;
    void *ctx;
} wubu_tool_t;

typedef struct {
    wubu_tool_t tools[WUBU_TOOL_MAX_TOOLS];
    int n_tools;
} wubu_tool_registry_t;

int wubu_tool_register(wubu_tool_registry_t *reg, const char *name,
                              const char *description,
                              const char *input_schema);
int wubu_tool_registry_init(wubu_tool_registry_t *reg);
int wubu_tool_set_handler(wubu_tool_registry_t *reg, const char *name,
                               wubu_tool_handler_t handler, void *ctx);
int wubu_tool_call(wubu_tool_registry_t *reg, const char *name,
                          const char *args_json,
                          char *result_buf, int buf_size);
int wubu_tool_parallel_call(wubu_tool_registry_t *reg,
                                   const char **names,
                                   const char **args, int n,
                                   char *results, int results_size);

#endif