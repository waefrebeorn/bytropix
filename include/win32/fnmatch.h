/* include/win32/fnmatch.h — minimal fnmatch shim for MSYS2/mingw64.
 * Supports the subset the engine uses: '*' wildcards and '?' single-char, plus
 * a literal match. Enough for shard glob "model-*-of-*.safetensors". */
#ifndef WUBU_WIN32_FNMATCH_H
#define WUBU_WIN32_FNMATCH_H

#define FNM_NOMATCH 1
#define FNM_PATHNAME 0x1
#define FNM_PERIOD   0x2
#define FNM_NOESCAPE 0x4

/* Returns 0 on match, FNM_NOMATCH otherwise. */
static inline int fnmatch(const char *pattern, const char *string, int flags) {
    (void)flags;
    const char *p = pattern, *s = string;
    while (*p) {
        if (*p == '*') {
            while (*p == '*') p++;
            if (*p == '\0') return 0;
            while (*s) {
                if (fnmatch(p, s, 0) == 0) return 0;
                s++;
            }
            return FNM_NOMATCH;
        } else if (*p == '?') {
            if (*s == '\0') return FNM_NOMATCH;
            p++; s++;
        } else {
            if (*s != *p) return FNM_NOMATCH;
            p++; s++;
        }
    }
    return (*s == '\0') ? 0 : FNM_NOMATCH;
}

#endif /* WUBU_WIN32_FNMATCH_H */
