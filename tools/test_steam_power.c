/* test_steam_power.c -- Steam Deck power management tests. */
#include "wubu_steam_power.h"
#include <stdio.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_steam_power_t sp;

    /* init: balanced, 100% battery, not charging, 45C */
    CHECK(wubu_sp_init(&sp) == 0);
    CHECK(sp.profile == WUBU_SP_PROFILE_BALANCED);
    CHECK(sp.tdp_w == 12);
    CHECK(sp.gpu_mhz_max == 1200);
    CHECK(sp.cpu_mhz_max == 2800);
    CHECK(sp.fan == WUBU_SP_FAN_BALANCED);
    CHECK(sp.battery_pct == 100);
    CHECK(sp.charging == 0);
    CHECK(sp.temp_c == 45);
    CHECK(wubu_sp_throttled(&sp) == 0);
    CHECK(sp.fan_rpm > 0);   /* balanced fan at 45C */

    /* preset table sanity */
    wubu_sp_preset_t p;
    CHECK(wubu_sp_preset(WUBU_SP_PROFILE_BATTERY_SAVER, &p) == 0);
    CHECK(p.tdp_w == 4 && p.gpu_mhz_max == 400 && p.fan == WUBU_SP_FAN_QUIET);
    CHECK(wubu_sp_preset(WUBU_SP_PROFILE_PERFORMANCE, &p) == 0);
    CHECK(p.tdp_w == 18 && p.gpu_mhz_max == 1600 && p.fan == WUBU_SP_FAN_AGGRESSIVE);
    CHECK(wubu_sp_preset((wubu_sp_profile_t)99, &p) == -1);
    CHECK(wubu_sp_preset(WUBU_SP_PROFILE_BALANCED, NULL) == -1);

    /* apply performance: raises TDP/GPU/fan */
    CHECK(wubu_sp_apply_profile(&sp, WUBU_SP_PROFILE_PERFORMANCE) == 0);
    CHECK(sp.tdp_w == 18);
    CHECK(sp.gpu_mhz_max == 1600);
    CHECK(sp.fan == WUBU_SP_FAN_AGGRESSIVE);
    CHECK(sp.n_profile_changes == 1);

    /* apply battery saver: clamps down */
    CHECK(wubu_sp_apply_profile(&sp, WUBU_SP_PROFILE_BATTERY_SAVER) == 0);
    CHECK(sp.tdp_w == 4);
    CHECK(sp.gpu_mhz_max == 400);
    CHECK(sp.fan == WUBU_SP_FAN_QUIET);
    CHECK(wubu_sp_apply_profile(&sp, (wubu_sp_profile_t)99) == -1);
    CHECK(wubu_sp_apply_profile(NULL, WUBU_SP_PROFILE_BALANCED) == -1);

    /* battery life: performance (18W) < balanced (12W) < saver (4W) */
    CHECK(wubu_sp_apply_profile(&sp, WUBU_SP_PROFILE_PERFORMANCE) == 0);
    int perf_min = wubu_sp_battery_minutes(&sp);
    CHECK(wubu_sp_apply_profile(&sp, WUBU_SP_PROFILE_BALANCED) == 0);
    int bal_min = wubu_sp_battery_minutes(&sp);
    CHECK(wubu_sp_apply_profile(&sp, WUBU_SP_PROFILE_BATTERY_SAVER) == 0);
    int sav_min = wubu_sp_battery_minutes(&sp);
    CHECK(perf_min < bal_min);
    CHECK(bal_min < sav_min);
    CHECK(sav_min > 0);
    /* on AC: not discharging */
    CHECK(wubu_sp_update(&sp, 50, 80, 1) == 0);
    CHECK(wubu_sp_battery_minutes(&sp) == -1);
    /* empty battery: 0 minutes */
    CHECK(wubu_sp_update(&sp, 50, 0, 0) == 0);
    CHECK(wubu_sp_battery_minutes(&sp) == 0);

    /* thermal throttle: >=95C clamps TDP/GPU/CPU */
    CHECK(wubu_sp_apply_profile(&sp, WUBU_SP_PROFILE_PERFORMANCE) == 0);
    CHECK(sp.tdp_w == 18);
    CHECK(wubu_sp_update(&sp, 96, 100, 1) == 0);   /* hot, on AC (no auto-clamp) */
    CHECK(wubu_sp_throttled(&sp) == 1);
    CHECK(sp.tdp_w == 10);            /* clamped from 18 */
    CHECK(sp.gpu_mhz_max == 600);     /* clamped from 1600 */
    CHECK(sp.cpu_mhz_max == 1800);    /* clamped from 3400 */
    /* cool down: next profile apply clears the clamp */
    CHECK(wubu_sp_update(&sp, 50, 100, 1) == 0);
    CHECK(wubu_sp_throttled(&sp) == 0);
    CHECK(wubu_sp_apply_profile(&sp, WUBU_SP_PROFILE_PERFORMANCE) == 0);
    CHECK(sp.tdp_w == 18);

    /* fan policies */
    CHECK(wubu_sp_set_fan(&sp, WUBU_SP_FAN_OFF) == 0);
    CHECK(sp.fan_rpm == 0);           /* off always */
    CHECK(wubu_sp_set_fan(&sp, WUBU_SP_FAN_QUIET) == 0);
    /* quiet at 50C is silent, at 80C spins */
    CHECK(wubu_sp_update(&sp, 50, 100, 1) == 0);
    CHECK(sp.fan_rpm == 0);
    CHECK(wubu_sp_update(&sp, 80, 100, 1) == 0);
    CHECK(sp.fan_rpm > 0);
    /* aggressive is louder than balanced at the same temp */
    CHECK(wubu_sp_set_fan(&sp, WUBU_SP_FAN_BALANCED) == 0);
    int bal_rpm = sp.fan_rpm;
    CHECK(wubu_sp_set_fan(&sp, WUBU_SP_FAN_AGGRESSIVE) == 0);
    CHECK(sp.fan_rpm > bal_rpm);
    CHECK(wubu_sp_set_fan(&sp, (wubu_sp_fan_t)99) == -1);
    CHECK(wubu_sp_set_fan(NULL, WUBU_SP_FAN_OFF) == -1);

    /* battery-saver auto-clamp: <15% on battery, unlocked -> battery saver */
    CHECK(wubu_sp_apply_profile(&sp, WUBU_SP_PROFILE_PERFORMANCE) == 0);
    CHECK(wubu_sp_set_locked(&sp, 0) == 0);
    CHECK(wubu_sp_update(&sp, 50, 10, 0) == 0);   /* 10% battery, discharging */
    CHECK(sp.profile == WUBU_SP_PROFILE_BATTERY_SAVER);
    CHECK(sp.tdp_w == 4);
    /* locked override blocks the auto-clamp */
    CHECK(wubu_sp_apply_profile(&sp, WUBU_SP_PROFILE_PERFORMANCE) == 0);
    CHECK(wubu_sp_set_locked(&sp, 1) == 0);
    CHECK(wubu_sp_update(&sp, 50, 10, 0) == 0);
    CHECK(sp.profile == WUBU_SP_PROFILE_PERFORMANCE);  /* stays */
    CHECK(sp.tdp_w == 18);
    /* charging blocks the auto-clamp too */
    CHECK(wubu_sp_set_locked(&sp, 0) == 0);
    CHECK(wubu_sp_apply_profile(&sp, WUBU_SP_PROFILE_PERFORMANCE) == 0);
    CHECK(wubu_sp_update(&sp, 50, 10, 1) == 0);   /* charging */
    CHECK(sp.profile == WUBU_SP_PROFILE_PERFORMANCE);

    /* bad args on update */
    CHECK(wubu_sp_update(&sp, 50, 101, 0) == -1);  /* battery > 100 */
    CHECK(wubu_sp_update(&sp, 50, -1, 0) == -1);
    CHECK(wubu_sp_update(NULL, 50, 50, 0) == -1);
    CHECK(wubu_sp_set_locked(NULL, 1) == -1);
    CHECK(wubu_sp_battery_minutes(NULL) == -1);
    CHECK(wubu_sp_throttled(NULL) == -1);

    if (fails == 0) printf("ALL STEAM_POWER TESTS PASSED\n");
    else printf("%d STEAM_POWER TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
