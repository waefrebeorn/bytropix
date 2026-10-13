/* wubu_dotfiles.c -- OMARCHY #279 dotfiles-as-config model (~/.config).
 * See wubu_dotfiles.h. XDG path resolution with fail-closed sandboxing. */
#include "wubu_dotfiles.h"
#include <string.h>
#include <stdio.h>

const char *wubu_xdg_base(wubu_xdg_kind_t kind) {
    switch (kind) {
        case WUBU_XDG_CONFIG: return ".config";
        case WUBU_XDG_DATA:   return ".local/share";
        case WUBU_XDG_CACHE:  return ".cache";
        case WUBU_XDG_STATE:  return ".local/state";
        default:              return ".config";
    }
}

int wubu_dotfiles_init(wubu_dotfiles_t *d, const char *root) {
    if (!d) return -1;
    if (!root || root[0] == '\0') {
        strncpy(d->root, "/", WUBU_XDG_PATH_MAX - 1);
    } else {
        size_t rl = strlen(root);
        if (rl >= WUBU_XDG_PATH_MAX) return -1;
        strncpy(d->root, root, WUBU_XDG_PATH_MAX - 1);
        /* strip a trailing slash (but keep a lone "/") */
        if (rl > 1 && d->root[rl - 1] == '/') d->root[rl - 1] = '\0';
    }
    d->root[WUBU_XDG_PATH_MAX - 1] = '\0';
    return 0;
}

/* Reject any component that is "." , ".." , empty, or contains a slash that
 * would traverse. app/relpath are single safe path segments. */
static int safe_segment(const char *s) {
    if (!s || s[0] == '\0') return 0;
    if (strcmp(s, ".") == 0 || strcmp(s, "..") == 0) return 0;
    for (const char *p = s; *p; p++) {
        if (*p == '/') return 0;      /* no nested traversal in a segment */
        if ((unsigned char)*p < 0x20) return 0;  /* no control chars */
    }
    return 1;
}

int wubu_dotfiles_path(wubu_dotfiles_t *d, wubu_xdg_kind_t kind,
                       const char *app, const char *relpath,
                       char *out, size_t outsz) {
    if (!d || !app || !out || outsz == 0) return -1;
    if (!safe_segment(app)) return -1;
    /* relpath may be NULL (just the app dir) or a single safe segment. */
    if (relpath && relpath[0] != '\0' && !safe_segment(relpath)) return -1;

    const char *base = wubu_xdg_base(kind);
    int n;
    if (relpath && relpath[0] != '\0')
        n = snprintf(out, outsz, "%s/%s/%s/%s", d->root, base, app, relpath);
    else
        n = snprintf(out, outsz, "%s/%s/%s", d->root, base, app);
    if (n < 0 || (size_t)n >= outsz) return -1;   /* truncation = fail-closed */
    return 0;
}

int wubu_dotfiles_within_root(const wubu_dotfiles_t *d, const char *path) {
    if (!d || !path) return 0;
    size_t rl = strlen(d->root);
    if (strncmp(path, d->root, rl) != 0) return 0;
    /* the char after the root prefix must be a slash or end (avoid /home/wubux) */
    if (path[rl] != '\0' && path[rl] != '/') return 0;
    /* no ".." component anywhere */
    const char *p = path;
    while (*p) {
        if (p[0] == '.' && p[1] == '.' && (p[2] == '/' || p[2] == '\0') &&
            (p == path || p[-1] == '/')) return 0;
        p++;
    }
    return 1;
}
