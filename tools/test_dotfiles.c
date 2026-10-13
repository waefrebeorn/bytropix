/* test_dotfiles.c -- OMARCHY #279 dotfiles-as-config model (~/.config).
 * Proves: config/data/cache/state resolve under the root, app dir with no
 * relpath, trailing-slash root normalised, ".."/absolute/segment-slash
 * escapes rejected (fail-closed), truncation rejected, within-root check. */
#include "wubu_dotfiles.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    char out[WUBU_XDG_PATH_MAX];
    wubu_dotfiles_t d;
    CHECK(wubu_dotfiles_init(&d, "/home/wubu") == 0, "init ok");

    /* XDG bases */
    CHECK(strcmp(wubu_xdg_base(WUBU_XDG_CONFIG), ".config") == 0, "config base = .config");
    CHECK(strcmp(wubu_xdg_base(WUBU_XDG_DATA), ".local/share") == 0, "data base = .local/share");
    CHECK(strcmp(wubu_xdg_base(WUBU_XDG_CACHE), ".cache") == 0, "cache base = .cache");
    CHECK(strcmp(wubu_xdg_base(WUBU_XDG_STATE), ".local/state") == 0, "state base = .local/state");

    /* config resolves under ~/.config */
    CHECK(wubu_dotfiles_path(&d, WUBU_XDG_CONFIG, "hypr", "hyprland.conf", out, sizeof out) == 0, "config path ok");
    CHECK(strcmp(out, "/home/wubu/.config/hypr/hyprland.conf") == 0, "config path value");

    /* app dir with no relpath */
    CHECK(wubu_dotfiles_path(&d, WUBU_XDG_CONFIG, "hypr", NULL, out, sizeof out) == 0, "app dir ok");
    CHECK(strcmp(out, "/home/wubu/.config/hypr") == 0, "app dir value");

    /* data/cache/state */
    CHECK(wubu_dotfiles_path(&d, WUBU_XDG_DATA, "app", "db.sqlite", out, sizeof out) == 0, "data path ok");
    CHECK(strcmp(out, "/home/wubu/.local/share/app/db.sqlite") == 0, "data path value");
    CHECK(wubu_dotfiles_path(&d, WUBU_XDG_CACHE, "app", "x", out, sizeof out) == 0, "cache path ok");
    CHECK(strcmp(out, "/home/wubu/.cache/app/x") == 0, "cache path value");

    /* trailing-slash root normalised */
    wubu_dotfiles_t d2;
    wubu_dotfiles_init(&d2, "/home/wubu/");
    CHECK(wubu_dotfiles_path(&d2, WUBU_XDG_CONFIG, "a", "b", out, sizeof out) == 0, "trailing-slash root ok");
    CHECK(strcmp(out, "/home/wubu/.config/a/b") == 0, "trailing-slash normalised (no double slash)");

    /* fail-closed: ".." escape in app or relpath */
    CHECK(wubu_dotfiles_path(&d, WUBU_XDG_CONFIG, "..", "x", out, sizeof out) == -1, "app '..' rejected");
    CHECK(wubu_dotfiles_path(&d, WUBU_XDG_CONFIG, "app", "..", out, sizeof out) == -1, "relpath '..' rejected");
    CHECK(wubu_dotfiles_path(&d, WUBU_XDG_CONFIG, ".", "x", out, sizeof out) == -1, "app '.' rejected");
    /* segment containing a slash (nested traversal) rejected */
    CHECK(wubu_dotfiles_path(&d, WUBU_XDG_CONFIG, "a/b", "x", out, sizeof out) == -1, "slash in segment rejected");
    CHECK(wubu_dotfiles_path(&d, WUBU_XDG_CONFIG, "app", "../../etc/passwd", out, sizeof out) == -1, "relpath traversal rejected");
    /* NULL / empty rejected */
    CHECK(wubu_dotfiles_path(&d, WUBU_XDG_CONFIG, "", "x", out, sizeof out) == -1, "empty app rejected");
    CHECK(wubu_dotfiles_path(&d, WUBU_XDG_CONFIG, "app", "x", NULL, 10) == -1, "NULL out rejected");

    /* truncation rejected (tiny buffer) */
    char tiny[8];
    CHECK(wubu_dotfiles_path(&d, WUBU_XDG_CONFIG, "hypr", "hyprland.conf", tiny, sizeof tiny) == -1, "truncation rejected");

    /* within-root check */
    CHECK(wubu_dotfiles_within_root(&d, "/home/wubu/.config/a/b") == 1, "within root true");
    CHECK(wubu_dotfiles_within_root(&d, "/etc/passwd") == 0, "outside root false");
    CHECK(wubu_dotfiles_within_root(&d, "/home/wubux/.config") == 0, "prefix-but-not-dir false");
    CHECK(wubu_dotfiles_within_root(&d, "/home/wubu/../etc") == 0, ".. escape false");
    CHECK(wubu_dotfiles_within_root(&d, NULL) == 0, "NULL path false");

    printf("\n=== test_dotfiles: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
