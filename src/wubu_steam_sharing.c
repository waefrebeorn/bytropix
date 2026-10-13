/* wubu_steam_sharing.c -- Steam Family Sharing.
 * See wubu_steam_sharing.h. Real, deterministic family-sharing gate. */

#include "wubu_steam_sharing.h"
#include <string.h>

int wubu_sh_init(wubu_steam_sharing_t *s, int max_borrowers,
                 int max_concurrent_per_game) {
    if (!s || max_borrowers < 0 || max_concurrent_per_game < 1) return -1;
    memset(s, 0, sizeof(*s));
    s->max_borrowers = max_borrowers;
    s->max_concurrent_per_game = max_concurrent_per_game;
    return 0;
}

static int find_game(const wubu_steam_sharing_t *s, const char *game) {
    if (!s || !game) return -1;
    for (int i = 0; i < s->n_games; i++)
        if (strcmp(s->games[i].game, game) == 0) return i;
    return -1;
}

static int find_borrower(const wubu_steam_sharing_t *s, const char *name) {
    if (!s || !name) return -1;
    for (int i = 0; i < s->n_borrowers; i++)
        if (strcmp(s->borrowers[i].name, name) == 0) return i;
    return -1;
}

int wubu_sh_add_game(wubu_steam_sharing_t *s, const char *game) {
    if (!s || !game) return -1;
    if (find_game(s, game) >= 0) return -1;   /* dup */
    if (s->n_games >= WUBU_SH_MAX_GAMES) return -1;
    wubu_sh_game_t *g = &s->games[s->n_games++];
    memset(g, 0, sizeof(*g));
    strncpy(g->game, game, WUBU_SH_NAME_MAX - 1);
    return 0;
}

int wubu_sh_share_game(wubu_steam_sharing_t *s, const char *game, int shared) {
    if (!s) return -1;
    int i = find_game(s, game);
    if (i < 0) return -1;
    s->games[i].shared = shared ? 1 : 0;
    return 0;
}

int wubu_sh_lender_play(wubu_steam_sharing_t *s, const char *game, int playing) {
    if (!s) return -1;
    int i = find_game(s, game);
    if (i < 0) return -1;
    s->games[i].lender_playing = playing ? 1 : 0;
    return 0;
}

int wubu_sh_authorize(wubu_steam_sharing_t *s, const char *borrower) {
    if (!s || !borrower) return -1;
    int i = find_borrower(s, borrower);
    if (i >= 0) { s->borrowers[i].authorized = 1; return 0; }   /* re-auth */
    if (s->n_borrowers >= WUBU_SH_MAX_BORROWERS) return -1;
    /* enforce the lender's borrower limit */
    int auth = 0;
    for (int k = 0; k < s->n_borrowers; k++) if (s->borrowers[k].authorized) auth++;
    if (auth >= s->max_borrowers) return -1;
    wubu_sh_borrower_t *b = &s->borrowers[s->n_borrowers++];
    memset(b, 0, sizeof(*b));
    strncpy(b->name, borrower, WUBU_SH_NAME_MAX - 1);
    b->authorized = 1;
    return 0;
}

int wubu_sh_revoke(wubu_steam_sharing_t *s, const char *borrower) {
    if (!s) return -1;
    int i = find_borrower(s, borrower);
    if (i < 0) return -1;
    /* kick any active session */
    if (s->borrowers[i].active_session) {
        int gi = find_game(s, s->borrowers[i].session_game);
        if (gi >= 0 && s->games[gi].n_active_borrowers > 0)
            s->games[gi].n_active_borrowers--;
        s->borrowers[i].active_session = 0;
        s->borrowers[i].session_game[0] = '\0';
    }
    s->borrowers[i].authorized = 0;
    return 0;
}

int wubu_sh_borrow_start(wubu_steam_sharing_t *s, const char *borrower,
                         const char *game) {
    if (!s || !borrower || !game) return -1;
    int bi = find_borrower(s, borrower);
    if (bi < 0) { s->n_denied++; return 0; }          /* unknown borrower */
    if (!s->borrowers[bi].authorized) { s->n_denied++; return 0; }
    if (s->borrowers[bi].active_session) { s->n_denied++; return 0; }  /* already in a session */
    int gi = find_game(s, game);
    if (gi < 0) { s->n_denied++; return 0; }          /* unknown game */
    if (!s->games[gi].shared) { s->n_denied++; return 0; }   /* not shared */
    if (s->games[gi].lender_playing) { s->n_denied++; return 0; }  /* lender has it open */
    if (s->games[gi].n_active_borrowers >= s->max_concurrent_per_game) {
        s->n_denied++; return 0;                       /* concurrency limit */
    }
    /* grant */
    s->games[gi].n_active_borrowers++;
    s->borrowers[bi].active_session = 1;
    strncpy(s->borrowers[bi].session_game, game, WUBU_SH_NAME_MAX - 1);
    return 1;
}

int wubu_sh_borrow_end(wubu_steam_sharing_t *s, const char *borrower) {
    if (!s) return -1;
    int bi = find_borrower(s, borrower);
    if (bi < 0 || !s->borrowers[bi].active_session) return -1;
    int gi = find_game(s, s->borrowers[bi].session_game);
    if (gi >= 0 && s->games[gi].n_active_borrowers > 0)
        s->games[gi].n_active_borrowers--;
    s->borrowers[bi].active_session = 0;
    s->borrowers[bi].session_game[0] = '\0';
    return 0;
}

int wubu_sh_is_shared(const wubu_steam_sharing_t *s, const char *game) {
    int i = find_game(s, game);
    if (i < 0) return -1;
    return s->games[i].shared ? 1 : 0;
}

int wubu_sh_n_active_borrowers(const wubu_steam_sharing_t *s, const char *game) {
    int i = find_game(s, game);
    if (i < 0) return -1;
    return s->games[i].n_active_borrowers;
}
