#ifndef WUBU_DOTFILES_H
#define WUBU_DOTFILES_H
#include <stddef.h>

/* wubu_dotfiles.h -- OMARCHY #279 dotfiles-as-config model (~/.config).
 *
 * Resolves all application configuration into a single, sandboxed base
 * (~/.config), following the XDG Base Directory convention but with a
 * fail-closed twist: every resolved path is validated to stay WITHIN the
 * configured root (no "..", no absolute escape, no symlink-style traversal).
 * This is the Arch/Omarchy "dotfiles-as-config" model: one root, all config,
 * nothing scattered. Deterministic, no allocation beyond the caller's
 * buffer, fail-closed (an escaping or oversized request yields an error).
 * Real work (no stubs). */

#define WUBU_XDG_PATH_MAX 512

typedef enum {
    WUBU_XDG_CONFIG=0,  /* ~/.config  -- editable config (the dotfiles) */
    WUBU_XDG_DATA,      /* ~/.local/share -- app data */
    WUBU_XDG_CACHE,     /* ~/.cache    -- regenerable cache */
    WUBU_XDG_STATE,     /* ~/.local/state -- logs, history */
} wubu_xdg_kind_t;

typedef struct {
    char root[WUBU_XDG_PATH_MAX];   /* sandbox root, e.g. "/home/wubu" */
} wubu_dotfiles_t;

/* Initialise with a sandbox root (e.g. "/home/wubu"). Empty root -> "/". */
int wubu_dotfiles_init(wubu_dotfiles_t *d, const char *root);

/* Resolve a config path: writes "<root>/<base>/<app>/<relpath>" into out
 * (size outsz). kind selects the XDG base. Returns 0 on success, -1 on
 * error (escape attempt, overflow, NULL). Fail-closed. */
int wubu_dotfiles_path(wubu_dotfiles_t *d, wubu_xdg_kind_t kind,
                       const char *app, const char *relpath,
                       char *out, size_t outsz);

/* The base subdirectory for a kind (".config", ".local/share", etc.). */
const char *wubu_xdg_base(wubu_xdg_kind_t kind);

/* Validate that a resolved path stays within the root (no escape). 1/0. */
int wubu_dotfiles_within_root(const wubu_dotfiles_t *d, const char *path);

#endif
