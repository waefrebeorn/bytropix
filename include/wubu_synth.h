/*
 * wubu_synth.h -- Program synthesis: spec→C11 code gen (AX05).
 */
#ifndef WUBU_SYNTH_H
#define WUBU_SYNTH_H

#define WUBU_SYNTH_MAX_NAME 64
#define WUBU_SYNTH_MAX_OUTPUT 8192

typedef struct {
    int n_templates;
    const char **templates;
    int n_generated;
} wubu_synth_t;

int wubu_synth_init(wubu_synth_t *s);
int wubu_synth_generate(const wubu_synth_t *s, int template_idx,
                                 const char *func_name,
                                 char *out, int out_size);
int wubu_synth_compile_verify(const char *source,
                                     const char *func_name);

#endif