/*
 * wubu_codesynth.c -- spec→C11 source generator (AX10). C11.
 *
 * Convergence (program synthesis 7-hop: HEURIGYM, Apeiron, EvoAgent, Code-as-Policies):
 *   - AX10: the agent receives a textual spec (operation + func name),
 *     generates C11 source from a template, compiles it, and reports.
 */
#include "wubu_codesynth.h"
#include "wubu_spawn.h"
#include <unistd.h>

static const char *gen_templates[] = {
    /* operation: "add" → int {name}(int a, int b) { return a + b; } */
    "int %s(int a, int b) { return a + b; }",
    /* "mul" → multiply */
    "int %s(int a, int b) { return a * b; }",
    /* "bufsum" → buffer sum */
    "int %s(const int *buf, int n) { int s=0; for(int i=0;i<n;i++) s+=buf[i]; return s; }",
    /* "max" → max of two */
    "int %s(int a, int b) { return a > b ? a : b; }",
    /* "min" → min of two */
    "int %s(int a, int b) { return a < b ? a : b; }",
};

int wubu_codesynth_init(wubu_codesynth_t *cs, const char *spec) {
    if (!cs || !spec) return -1;
    snprintf(cs->spec, sizeof(cs->spec), "%s", spec);
    cs->source[0] = '\0';
    cs->compiled = 0;
    cs->verified = 0;
    return 0;
}

static int match_op(const char *op, const char *keyword) {
    if (!op || !keyword) return 0;
    return (strstr(op, keyword) != NULL) ? 1 : 0;
}

int wubu_codesynth_generate(wubu_codesynth_t *cs, const char *func_name,
                            const char *operation, char *out, int out_size) {
    if (!cs || !func_name || !operation || !out || out_size <= 0) return -1;
    int idx = -1;
    if (match_op(operation, "add") || match_op(operation, "sum of")) idx = 0;
    else if (match_op(operation, "mul") || match_op(operation, "multiply")) idx = 1;
    else if (match_op(operation, "buf") || match_op(operation, "buffer")) idx = 2;
    else if (match_op(operation, "max")) idx = 3;
    else if (match_op(operation, "min")) idx = 4;
    else idx = 0;  /* default: add */

    const char *tmpl = gen_templates[idx];
    snprintf(out, out_size, tmpl, func_name);
    snprintf(cs->source, sizeof(cs->source), "%s", out);
    return 0;
}

int wubu_codesynth_compile(const wubu_codesynth_t *cs, const char *src) {
    if (!cs || !src) return -1;
    const char *path = "/tmp/wubu_codesynth_tmp.c";
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "#include <stdio.h>\n%s\nint main(){printf(\"%%d\\n\", %s(2,3)); return 0;}",
            src, strstr(cs->spec, "func:") ? "generated_func" : "generated_func");
    fclose(f);
    char *argv[] = { "gcc", "-O2", "-Wall", "-o", "/tmp/wubu_codesynth_tmp",
                     (char *)path, NULL };
    int rc = wubu_spawn_wait("gcc", (char *const *)argv, true);
    unlink(path);
    unlink("/tmp/wubu_codesynth_tmp");
    return rc == 0 ? 1 : 0;
}