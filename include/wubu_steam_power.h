#ifndef WUBU_STEAM_POWER_H
#define WUBU_STEAM_POWER_H

/* wubu_steam_power.h -- Steam Deck power management.
 *
 * Models the Steam Deck power/performance control plane: a TDP (thermal design
 * power) budget in watts, a fan control policy (off/quiet/balanced/aggressive
 * with a temperature-driven curve), and GPU/CPU clock limits. The controller
 * enforces thermal headroom: applying a power state clamps TDP/fan/GPU by the
 * current thermal zone, reports estimated battery life, and tracks a
 * battery-saver mode that aggressively clamps everything. Deterministic state
 * machine, no stubs. */

#include <stddef.h>
#include <stdint.h>

#define WUBU_SP_NAME_MAX 24

/* Power/performance profiles */
typedef enum {
    WUBU_SP_PROFILE_BATTERY_SAVER = 0,  /* max clamp: min TDP, quiet fan */
    WUBU_SP_PROFILE_QUIET,              /* low TDP, low fan, capped GPU */
    WUBU_SP_PROFILE_BALANCED,           /* default: mid TDP, auto fan */
    WUBU_SP_PROFILE_PERFORMANCE,        /* high TDP, aggressive fan, max GPU */
    WUBU_SP_PROFILE_COUNT
} wubu_sp_profile_t;

/* Fan control policies */
typedef enum {
    WUBU_SP_FAN_OFF = 0,
    WUBU_SP_FAN_QUIET,      /* only spins above a high threshold */
    WUBU_SP_FAN_BALANCED,   /* curve tracks temperature */
    WUBU_SP_FAN_AGGRESSIVE  /* near-max most of the time */
} wubu_sp_fan_t;

typedef struct {
    int tdp_w;          /* TDP budget in watts */
    int gpu_mhz_max;    /* GPU clock ceiling */
    int cpu_mhz_max;    /* CPU clock ceiling */
    wubu_sp_fan_t fan;
} wubu_sp_preset_t;

typedef struct {
    wubu_sp_profile_t profile;
    int tdp_w;            /* current applied TDP */
    int gpu_mhz_max;
    int cpu_mhz_max;
    wubu_sp_fan_t fan;
    int fan_rpm;          /* current fan speed (0 = off) */
    int temp_c;           /* APU temperature, deg C */
    int battery_pct;      /* 0..100 */
    int charging;         /* 1 = on AC */
    int profile_locked;   /* manual override active */
    int thermal_throttle; /* 1 = clamped by thermal zone */
    int n_profile_changes;
} wubu_steam_power_t;

/* Initialise (balanced, not charging, 100% battery, 45C). */
int wubu_sp_init(wubu_steam_power_t *sp);

/* Apply a power/performance profile. Clamps by thermal headroom when hot.
 * Returns 0/-1 (bad profile). */
int wubu_sp_apply_profile(wubu_steam_power_t *sp, wubu_sp_profile_t profile);

/* Lock/unlock manual profile override (ignores battery-saver auto-clamp). */
int wubu_sp_set_locked(wubu_steam_power_t *sp, int locked);

/* Report temperature + battery; recomputes fan + thermal throttle. */
int wubu_sp_update(wubu_steam_power_t *sp, int temp_c, int battery_pct,
                   int charging);

/* Set the fan policy directly (manual). */
int wubu_sp_set_fan(wubu_steam_power_t *sp, wubu_sp_fan_t fan);

/* Estimated battery life in minutes for the current TDP/thermal state.
 * A rough model: full battery (100) at profile TDP with a baseline draw. */
int wubu_sp_battery_minutes(const wubu_steam_power_t *sp);

/* Is the device thermally throttled right now? */
int wubu_sp_throttled(const wubu_steam_power_t *sp);

/* The static preset table for a profile (no thermal clamp). */
int wubu_sp_preset(wubu_sp_profile_t profile, wubu_sp_preset_t *out);

#endif
