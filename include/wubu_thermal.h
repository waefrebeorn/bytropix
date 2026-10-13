#ifndef WUBU_THERMAL_H
#define WUBU_THERMAL_H

/* wubu_thermal.h -- KERNEL #32 thermal governor set via /n/hw/<dev>/thermal.
 *
 * A per-device thermal governor exposed through the 9P namespace as a
 * writable file `thermal` under /n/hw/<dev>/. Writing a governor name selects
 * the policy; reading returns the current governor + last temperature. The
 * governor decides throttling from the device temperature against per-mode
 * thresholds: crossing a threshold trips a throttle level that the driver
 * layer acts on. Unknown governors and out-of-range temperatures fail-closed
 * (keep the current safe governor). Deterministic, real work (no stubs). */

#include <stddef.h>
#include <stdint.h>

#define WUBU_THERMAL_MAX_DEVS 16
#define WUBU_THERMAL_NAME_MAX 48

typedef enum {
    WUBU_THERM_PERFORMANCE = 0,  /* max clocks, throttle only near critical */
    WUBU_THERM_BALANCED    = 1,  /* default: throttle earlier */
    WUBU_THERM_POWERSAVE   = 2   /* throttle aggressively, cool-first */
} wubu_thermal_gov_t;

typedef enum {
    WUBU_THERM_NONE     = 0,   /* no throttle */
    WUBU_THERM_LIGHT    = 1,   /* mild clock reduction */
    WUBU_THERM_HEAVY    = 2,   /* strong clock reduction */
    WUBU_THERM_CRITICAL = 3    /* emergency: park the device */
} wubu_thermal_level_t;

typedef struct {
    char name[WUBU_THERMAL_NAME_MAX];   /* device name, e.g. "gpu0", "cpu0" */
    int present;
    wubu_thermal_gov_t governor;
    int temp_mc;                 /* last temperature, milli-degrees Celsius */
    int throttle;                /* current throttle level (wubu_thermal_level_t) */
    /* per-governor throttle trip points (milli-degrees C) */
    int light_mc, heavy_mc, critical_mc;
} wubu_thermal_dev_t;

typedef struct {
    wubu_thermal_dev_t devs[WUBU_THERMAL_MAX_DEVS];
    int n;
} wubu_thermal_t;

/* Initialise the thermal registry. Returns 0. */
int wubu_thermal_init(wubu_thermal_t *t);

/* Register a device with a starting governor. Thresholds are derived from the
 * governor. Returns 0/-1. */
int wubu_thermal_add(wubu_thermal_t *t, const char *name, wubu_thermal_gov_t gov);

/* Find a device index by name, or -1. */
int wubu_thermal_find(const wubu_thermal_t *t, const char *name);

/* Set the governor via the /n write path. Unknown governor -> keep current,
 * return -1 (fail-closed). Returns 0 on change. */
int wubu_thermal_set_gov(wubu_thermal_t *t, const char *name, wubu_thermal_gov_t gov);

/* Parse a governor name ("performance"/"balanced"/"powersave"). Returns the
 * enum, or -1 if unrecognised. */
int wubu_thermal_gov_from_name(const char *name);

/* Governor name for an enum (never NULL). */
const char *wubu_thermal_gov_name(wubu_thermal_gov_t gov);

/* Update a device's temperature (milli-C) and recompute the throttle level
 * from the current governor's thresholds. Returns the new throttle level, or
 * -1 if the device is unknown. Out-of-range temp (< -100000 or > 200000) is
 * rejected (fail-closed: keeps prior temp/throttle). */
int wubu_thermal_update(wubu_thermal_t *t, const char *name, int temp_mc);

/* Current throttle level for a device, or -1. */
int wubu_thermal_throttle(const wubu_thermal_t *t, const char *name);

#endif
