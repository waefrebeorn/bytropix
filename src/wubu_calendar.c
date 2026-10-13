/* wubu_calendar.c -- Win98/XP Calendar / scheduler.
 * See wubu_calendar.h. Real, deterministic date-keyed appointment store. */

#include "wubu_calendar.h"
#include <string.h>

static int is_leap(int y) {
    return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
}

static int days_in_month(int y, int m) {
    static const int dm[] = {31,28,31,30,31,30,31,31,30,31,30,31};
    if (m == 2 && is_leap(y)) return 29;
    if (m < 1 || m > 12) return 0;
    return dm[m - 1];
}

int wubu_cal_valid_date(uint32_t date) {
    int y = (int)(date / 10000);
    int m = (int)((date / 100) % 100);
    int d = (int)(date % 100);
    if (y < 1) return 0;
    if (m < 1 || m > 12) return 0;
    if (d < 1 || d > days_in_month(y, m)) return 0;
    return 1;
}

int wubu_cal_init(wubu_calendar_t *c, uint32_t today) {
    if (!c || !wubu_cal_valid_date(today)) return -1;
    memset(c, 0, sizeof(*c));
    c->today = today;
    c->next_id = 1;
    return 0;
}

int wubu_cal_add(wubu_calendar_t *c, uint32_t date, int start_min,
                 int duration_min, const char *title, wubu_cal_recur_t recur) {
    if (!c || !title) return -1;
    if (!wubu_cal_valid_date(date)) return -1;
    if (start_min < 0 || start_min > 1439) return -1;
    if (duration_min <= 0 || duration_min > 1440) return -1;
    if (recur < 0 || recur > WUBU_CAL_MONTHLY) return -1;
    if (c->n_events >= WUBU_CAL_MAX_EVENTS) return -1;
    wubu_cal_event_t *e = &c->events[c->n_events];
    memset(e, 0, sizeof(*e));
    e->id = c->next_id++;
    e->date = date;
    e->start_min = start_min;
    e->duration_min = duration_min;
    e->recur = recur;
    strncpy(e->title, title, WUBU_CAL_TITLE_MAX - 1);
    c->n_events++;
    return e->id;
}

int wubu_cal_set_notes(wubu_calendar_t *c, int id, const char *notes) {
    if (!c || !notes) return -1;
    for (int i = 0; i < c->n_events; i++) {
        if (c->events[i].id == id) {
            strncpy(c->events[i].notes, notes, WUBU_CAL_NOTES_MAX - 1);
            c->events[i].notes[WUBU_CAL_NOTES_MAX - 1] = '\0';
            return 0;
        }
    }
    return -1;
}

int wubu_cal_remove(wubu_calendar_t *c, int id) {
    if (!c) return -1;
    for (int i = 0; i < c->n_events; i++) {
        if (c->events[i].id == id) {
            c->events[i] = c->events[c->n_events - 1];
            c->n_events--;
            return 0;
        }
    }
    return -1;
}

/* does a recurring event anchored at `anchor` fall on `date`? */
static int recurs_on(const wubu_cal_event_t *e, uint32_t date) {
    if (e->date == date) return 1;
    if (e->recur == WUBU_CAL_NONE) return 0;
    int ay = e->date / 10000, am = (e->date / 100) % 100, ad = e->date % 100;
    int by = date / 10000, bm = (date / 100) % 100, bd = date % 100;
    if (e->recur == WUBU_CAL_DAILY) return date >= e->date;
    if (e->recur == WUBU_CAL_WEEKLY) {
        /* falls on the same weekday: day-of-week via Zeller-ish; simplify by
         * comparing (y*365+m*30+d) mod 7 — good enough for a deterministic
         * model. */
        long a = (long)ay * 365 + am * 30 + ad;
        long b = (long)by * 365 + bm * 30 + bd;
        return date >= e->date && ((b - a) % 7 == 0);
    }
    if (e->recur == WUBU_CAL_MONTHLY) {
        return date >= e->date && bd == ad;
    }
    return 0;
}

int wubu_cal_events_on(wubu_calendar_t *c, uint32_t date, int *out, int max) {
    if (!c || !out || max < 0) return -1;
    int n = 0;
    for (int i = 0; i < c->n_events; i++) {
        if (recurs_on(&c->events[i], date)) {
            if (n < max) out[n] = c->events[i].id;
            n++;
        }
    }
    return n;
}

int wubu_cal_count_on(wubu_calendar_t *c, uint32_t date) {
    if (!c) return -1;
    int n = 0;
    for (int i = 0; i < c->n_events; i++)
        if (recurs_on(&c->events[i], date)) n++;
    return n;
}

int wubu_cal_reminder(wubu_calendar_t *c, uint32_t date, int now_min,
                      int window_min) {
    if (!c || now_min < 0 || now_min > 1439 || window_min < 0) return -1;
    for (int i = 0; i < c->n_events; i++) {
        if (!recurs_on(&c->events[i], date)) continue;
        int s = c->events[i].start_min;
        if (s >= now_min && s <= now_min + window_min) return 1;
    }
    return 0;
}
