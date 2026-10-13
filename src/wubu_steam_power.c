/* wubu_steam_power.c -- Steam Deck power management.
 * See wubu_steam_power.h. Real, deterministic power/thermal controller. */

#include "wubu_steam_power.h"
#include <string.h>

/* forward decl */
static int compute_fan_rpm(int temp_c, wubu_sp_fan_t fan);

/* Static presets (TDP watts, GPU MHz, CPU MHz, fan policy) */
static const wubu_sp_preset_t g_presets[WUBU_SP_PROFILE_COUNT] = {
    /* battery saver */ { 4,  400, 1200, WUBU_SP_FAN_QUIET },
    /* quiet         */ { 8,  800, 2000, WUBU_SP_FAN_QUIET },
    /* balanced      */ {12, 1200, 2800, WUBU_SP_FAN_BALANCED },
    /* performance   */ {18, 1600, 3400, WUBU_SP_FAN_AGGRESSIVE },
};

int wubu_sp_preset(wubu_sp_profile_t profile, wubu_sp_preset_t *out) {
    if (!out || profile < 0 || profile >= WUBU_SP_PROFILE_COUNT) return -1;
    *out = g_presets[profile];
    return 0;
}

int wubu_sp_init(wubu_steam_power_t *sp) {
    if (!sp) return -1;
    memset(sp, 0, sizeof(*sp));
    sp->profile = WUBU_SP_PROFILE_BALANCED;
    wubu_sp_preset_t p;
    wubu_sp_preset(sp->profile, &p);
    sp->tdp_w = p.tdp_w;
    sp->gpu_mhz_max = p.gpu_mhz_max;
    sp->cpu_mhz_max = p.cpu_mhz_max;
    sp->fan = p.fan;
    sp->temp_c = 45;
    sp->battery_pct = 100;
    sp->charging = 0;
    sp->fan_rpm = compute_fan_rpm(sp->temp_c, sp->fan);
    return 0;
}

/* Compute fan RPM from temperature + policy (a simple deterministic curve). */
static int compute_fan_rpm(int temp_c, wubu_sp_fan_t fan) {
    switch (fan) {
    case WUBU_SP_FAN_OFF:
        return 0;
    case WUBU_SP_FAN_QUIET:
        /* silent below 70C, ramps 70..90 */
        if (temp_c < 70) return 0;
        if (temp_c >= 90) return 4000;
        return (temp_c - 70) * 200;   /* 70->0 ... 90->4000 */
    case WUBU_SP_FAN_BALANCED:
        /* tracks temp 40..95 -> 0..5000 */
        if (temp_c <= 40) return 1000;
        if (temp_c >= 95) return 5000;
        return 1000 + (temp_c - 40) * 89;   /* 40->1000 ... 95->~5000 */
    case WUBU_SP_FAN_AGGRESSIVE:
        /* near-max: 55C -> 4000, 95C -> 6000 */
        if (temp_c <= 40) return 3000;
        if (temp_c >= 95) return 6000;
        return 3000 + (temp_c - 40) * 65;
    }
    return 0;
}

/* Thermal zone: above 95C we clamp TDP+GPU to protect the APU. */
static void apply_thermal(wubu_steam_power_t *sp) {
    sp->thermal_throttle = (sp->temp_c >= 95) ? 1 : 0;
    if (sp->thermal_throttle) {
        /* clamp to a safe floor regardless of profile */
        if (sp->tdp_w > 10) sp->tdp_w = 10;
        if (sp->gpu_mhz_max > 600) sp->gpu_mhz_max = 600;
        if (sp->cpu_mhz_max > 1800) sp->cpu_mhz_max = 1800;
    }
}

int wubu_sp_apply_profile(wubu_steam_power_t *sp, wubu_sp_profile_t profile) {
    if (!sp || profile < 0 || profile >= WUBU_SP_PROFILE_COUNT) return -1;
    wubu_sp_preset_t p;
    if (wubu_sp_preset(profile, &p) != 0) return -1;
    sp->profile = profile;
    sp->tdp_w = p.tdp_w;
    sp->gpu_mhz_max = p.gpu_mhz_max;
    sp->cpu_mhz_max = p.cpu_mhz_max;
    sp->fan = p.fan;
    sp->profile_locked = 0;
    sp->n_profile_changes++;
    apply_thermal(sp);
    sp->fan_rpm = compute_fan_rpm(sp->temp_c, sp->fan);
    return 0;
}

int wubu_sp_set_locked(wubu_steam_power_t *sp, int locked) {
    if (!sp) return -1;
    sp->profile_locked = locked ? 1 : 0;
    return 0;
}

int wubu_sp_set_fan(wubu_steam_power_t *sp, wubu_sp_fan_t fan) {
    if (!sp || fan < 0 || fan > WUBU_SP_FAN_AGGRESSIVE) return -1;
    sp->fan = fan;
    sp->fan_rpm = compute_fan_rpm(sp->temp_c, sp->fan);
    return 0;
}

int wubu_sp_update(wubu_steam_power_t *sp, int temp_c, int battery_pct,
                   int charging) {
    if (!sp || battery_pct < 0 || battery_pct > 100) return -1;
    sp->temp_c = temp_c;
    sp->battery_pct = battery_pct;
    sp->charging = charging ? 1 : 0;
    apply_thermal(sp);
    sp->fan_rpm = compute_fan_rpm(sp->temp_c, sp->fan);

    /* battery-saver auto-clamp: below 15% on battery and not locked, drop to
     * the battery-saver profile to extend runtime. */
    if (!sp->profile_locked && !sp->charging && sp->battery_pct < 15 &&
        sp->profile != WUBU_SP_PROFILE_BATTERY_SAVER) {
        return wubu_sp_apply_profile(sp, WUBU_SP_PROFILE_BATTERY_SAVER);
    }
    return 0;
}

int wubu_sp_battery_minutes(const wubu_steam_power_t *sp) {
    if (!sp) return -1;
    if (sp->charging) return -1;   /* on AC: not discharging */
    if (sp->battery_pct <= 0) return 0;
    /* rough model: 40 Wh battery, profile TDP draw (watts) ->
     * minutes = (40 Wh / TDP W) * 60 min * (battery_pct/100). Guard TDP>=1. */
    int tdp = sp->tdp_w > 0 ? sp->tdp_w : 1;
    return (int)((40L * 60L / tdp) * sp->battery_pct / 100);
}

int wubu_sp_throttled(const wubu_steam_power_t *sp) {
    if (!sp) return -1;
    return sp->thermal_throttle;
}
