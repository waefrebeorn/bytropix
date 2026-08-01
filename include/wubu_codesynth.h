#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- wubu_codesynth.h ---- */
#ifndef WUBU_CODESYNTH_H
#define WUBU_CODESYNTH_H
#define WUBU_SYN_MAX_SPEC 2048
#define WUBU_SYN_MAX_SRC 8192
typedef struct {
    char spec[WUBU_SYN_MAX_SPEC];
    char source[WUBU_SYN_MAX_SRC];
    int compiled;
    int verified;
} wubu_codesynth_t;

int  wubu_codesynth_init(wubu_codesynth_t *cs, const char *spec);
int  wubu_codesynth_generate(wubu_codesynth_t *cs, const char *func_name,
                             const char *operation, char *out, int out_size);
int  wubu_codesynth_compile(const wubu_codesynth_t *cs, const char *src);
#endif
