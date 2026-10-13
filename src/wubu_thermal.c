/* wubu_thermal.c -- KERNEL #32 thermal governor set via /n/hw/<dev>/thermal.
 * See wubu_thermal.h. Real, deterministic implementation. */

#include "wubu_thermal.h"
#include <string.h>

const char *wubu_thermal_gov_name(wubu_thermal_gov_t gov)
{
    switch (gov) {
    case WUBU_THERM_PERFORMANCE: return "performance";
    case WUBU_THERM_BALANCED:    return "balanced";
    case WUBU_THERM_POWERSAVE:   return "powersave";
    default:                     return "unknown";
    }
}

int wubu_thermal_gov_from_name(const char *name)
{
    if (!name) return -1;
    if (strcmp(name, "performance") == 0) return WUBU_THERM_PERFORMANCE;
    if (strcmp(name, "balanced") == 0)    return WUBU_THERM_BALANCED;
    if (strcmp(name, "powersave") == 0)   return WUBU_THERM_POWERSAVE;
    return -1;
}

/* Threshold trip points (milli-C) per governor. */
static void thresholds_for(wubu_thermal_gov_t gov, int *light, int *heavy, int *crit)
{
    switch (gov) {
    case WUBU_THERM_PERFORMANCE:
        *light = 90000; *heavy = 100000; *crit = 110000; break;
    case WUBU_THERM_BALANCED:
        *light = 75000; *heavy = 90000;  *crit = 100000; break;
    case WUBU_THERM_POWERSAVE:
    default:
        *light = 60000; *heavy = 75000;  *crit = 90000;  break;
    }
}

int wubu_thermal_init(wubu_thermal_t *t)
{
    if (!t) return -1;
    memset(t, 0, sizeof(*t));
    return 0;
}

int wubu_thermal_add(wubu_thermal_t *t, const char *name, wubu_thermal_gov_t gov)
{
    if (!t || !name) return -1;
    if (gov < WUBU_THERM_PERFORMANCE || gov > WUBU_THERM_POWERSAVE) return -1;
    if (wubu_thermal_find(t, name) >= 0) return -1;
    if (t->n >= WUBU_THERMAL_MAX_DEVS) return -1;
    wubu_thermal_dev_t *d = &t->devs[t->n++];
    memset(d, 0, sizeof(*d));
    strncpy(d->name, name, WUBU_THERMAL_NAME_MAX - 1);
    d->present = 1;
    d->governor = gov;
    d->temp_mc = 25000;   /* sane ambient default */
    d->throttle = WUBU_THERM_NONE;
    thresholds_for(gov, &d->light_mc, &d->heavy_mc, &d->critical_mc);
    return 0;
}

int wubu_thermal_find(const wubu_thermal_t *t, const char *name)
{
    if (!t || !name) return -1;
    for (int i = 0; i < t->n; i++)
        if (t->devs[i].present && strcmp(t->devs[i].name, name) == 0)
            return i;
    return -1;
}

int wubu_thermal_set_gov(wubu_thermal_t *t, const char *name, wubu_thermal_gov_t gov)
{
    if (!t) return -1;
    int i = wubu_thermal_find(t, name);
    if (i < 0) return -1;
    if (gov < WUBU_THERM_PERFORMANCE || gov > WUBU_THERM_POWERSAVE) return -1;
    wubu_thermal_dev_t *d = &t->devs[i];
    d->governor = gov;
    thresholds_for(gov, &d->light_mc, &d->heavy_mc, &d->critical_mc);
    /* recompute throttle under the new thresholds (ignore its level return) */
    wubu_thermal_update(t, name, d->temp_mc);
    return 0;
}

int wubu_thermal_update(wubu_thermal_t *t, const char *name, int temp_mc)
{
    if (!t) return -1;
    int i = wubu_thermal_find(t, name);
    if (i < 0) return -1;
    /* fail-closed on implausible temperatures */
    if (temp_mc < -100000 || temp_mc > 200000) return t->devs[i].throttle;
    wubu_thermal_dev_t *d = &t->devs[i];
    d->temp_mc = temp_mc;
    int lvl;
    if (temp_mc >= d->critical_mc)      lvl = WUBU_THERM_CRITICAL;
    else if (temp_mc >= d->heavy_mc)    lvl = WUBU_THERM_HEAVY;
    else if (temp_mc >= d->light_mc)    lvl = WUBU_THERM_LIGHT;
    else                                lvl = WUBU_THERM_NONE;
    d->throttle = lvl;
    return lvl;
}

int wubu_thermal_throttle(const wubu_thermal_t *t, const char *name)
{
    int i = wubu_thermal_find(t, name);
    return i < 0 ? -1 : t->devs[i].throttle;
}
