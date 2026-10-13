#ifndef WUBU_STEAM_SHARING_H
#define WUBU_STEAM_SHARING_H

/* wubu_steam_sharing.h -- Steam Family Sharing.
 *
 * Models Steam Family Library Sharing: a lender authorizes up to N devices/
 * accounts to borrow their library; a borrower can play a shared game only if
 * (a) the lender has authorized them, (b) the lender is not currently playing
 * that game (exclusive access), and (c) the shared-game concurrency limit is
 * not exceeded. Tracks per-game share grants, borrower sessions, and the
 * exclusivity lock. Deterministic, no stubs. */

#include <stddef.h>
#include <stdint.h>

#define WUBU_SH_MAX_GAMES 64
#define WUBU_SH_MAX_BORROWERS 16
#define WUBU_SH_NAME_MAX 48

typedef struct {
    char game[WUBU_SH_NAME_MAX];
    int shared;              /* lender authorized sharing for this game */
    int lender_playing;      /* lender is currently playing it (blocks borrow) */
    int n_active_borrowers;  /* concurrent borrowers of this game */
} wubu_sh_game_t;

typedef struct {
    char name[WUBU_SH_NAME_MAX];
    int authorized;          /* authorized to borrow the library */
    int active_session;      /* currently has a shared game open */
    char session_game[WUBU_SH_NAME_MAX];
} wubu_sh_borrower_t;

typedef struct {
    wubu_sh_game_t games[WUBU_SH_MAX_GAMES];
    int n_games;
    wubu_sh_borrower_t borrowers[WUBU_SH_MAX_BORROWERS];
    int n_borrowers;
    int max_borrowers;       /* lender's device/account limit */
    int max_concurrent_per_game;  /* concurrent borrowers allowed per game */
    int n_denied;            /* borrow attempts denied */
} wubu_steam_sharing_t;

/* Initialise (max_borrowers, max_concurrent_per_game). */
int wubu_sh_init(wubu_steam_sharing_t *s, int max_borrowers,
                 int max_concurrent_per_game);

/* Add a game to the lender's library (shared=0 by default). */
int wubu_sh_add_game(wubu_steam_sharing_t *s, const char *game);

/* Authorize sharing of a game. */
int wubu_sh_share_game(wubu_steam_sharing_t *s, const char *game, int shared);

/* The lender starts/stops playing a game (drives the exclusivity lock). */
int wubu_sh_lender_play(wubu_steam_sharing_t *s, const char *game, int playing);

/* Authorize a borrower account. */
int wubu_sh_authorize(wubu_steam_sharing_t *s, const char *borrower);

/* Revoke a borrower's authorization (kicks any active session). */
int wubu_sh_revoke(wubu_steam_sharing_t *s, const char *borrower);

/* A borrower requests to start a shared game. Returns 1 (granted, session
 * opened), 0 (denied), -1 (bad args). */
int wubu_sh_borrow_start(wubu_steam_sharing_t *s, const char *borrower,
                         const char *game);

/* A borrower ends their session. */
int wubu_sh_borrow_end(wubu_steam_sharing_t *s, const char *borrower);

/* Query. */
int wubu_sh_is_shared(const wubu_steam_sharing_t *s, const char *game);
int wubu_sh_n_active_borrowers(const wubu_steam_sharing_t *s, const char *game);

#endif
