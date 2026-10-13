#ifndef WUBU_CALENDAR_H
#define WUBU_CALENDAR_H

/* wubu_calendar.h -- Win98/XP Calendar / scheduler.
 *
 * A date-keyed appointment store: appointments live on a day (encoded as an
 * 8-digit YYYYMMDD integer), carry a start time (minutes from midnight),
 * duration, optional title + notes, and a recurrence rule (none / daily /
 * weekly / monthly). Supports add/remove, day lookup, and a reminder query
 * (appointments starting within a window from a given minute-of-day).
 * Deterministic, no stubs. */

#include <stddef.h>
#include <stdint.h>

#define WUBU_CAL_MAX_EVENTS 128
#define WUBU_CAL_TITLE_MAX 48
#define WUBU_CAL_NOTES_MAX 128

typedef enum {
    WUBU_CAL_NONE = 0,
    WUBU_CAL_DAILY,
    WUBU_CAL_WEEKLY,
    WUBU_CAL_MONTHLY
} wubu_cal_recur_t;

typedef struct {
    uint32_t date;        /* YYYYMMDD */
    int start_min;        /* minutes from midnight (0..1439) */
    int duration_min;     /* >0 */
    char title[WUBU_CAL_TITLE_MAX];
    char notes[WUBU_CAL_NOTES_MAX];
    wubu_cal_recur_t recur;
    int id;
} wubu_cal_event_t;

typedef struct {
    wubu_cal_event_t events[WUBU_CAL_MAX_EVENTS];
    int n_events;
    int next_id;
    uint32_t today;       /* current day (YYYYMMDD) for reminder windowing */
} wubu_calendar_t;

/* Initialise (today = given date). Returns 0/-1. */
int wubu_cal_init(wubu_calendar_t *c, uint32_t today);

/* Validate a YYYYMMDD date (1 <= month <= 12, 1 <= day <= days-in-month,
 * year >= 1). Returns 1/0. Leap years handled. */
int wubu_cal_valid_date(uint32_t date);

/* Add an appointment. Returns the event id, -1 on bad args (invalid date,
 * start/duration out of range) or full. */
int wubu_cal_add(wubu_calendar_t *c, uint32_t date, int start_min,
                 int duration_min, const char *title, wubu_cal_recur_t recur);

/* Set/append notes to an event by id. Returns 0/-1. */
int wubu_cal_set_notes(wubu_calendar_t *c, int id, const char *notes);

/* Remove an event by id. Returns 0/-1. */
int wubu_cal_remove(wubu_calendar_t *c, int id);

/* Count events on a given day (including recurrences that fall on it). */
int wubu_cal_count_on(wubu_calendar_t *c, uint32_t date);

/* Fill `out` with event ids on a given day (including recurrences), up to
 * `max`. Returns the count (may exceed max if truncated), -1 on bad args. */
int wubu_cal_events_on(wubu_calendar_t *c, uint32_t date, int *out, int max);

/* Reminder: does any event (on `date`) start within `window_min` minutes
 * after `now_min`? Returns 1 (reminder due), 0 (none), -1 bad args. */
int wubu_cal_reminder(wubu_calendar_t *c, uint32_t date, int now_min,
                      int window_min);

#endif
