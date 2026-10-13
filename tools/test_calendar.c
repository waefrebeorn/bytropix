/* test_calendar.c -- Win98/XP Calendar / scheduler tests. */
#include "wubu_calendar.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_calendar_t c;

    /* date validation */
    CHECK(wubu_cal_valid_date(20260229) == 0);   /* 2026 not a leap year */
    CHECK(wubu_cal_valid_date(20240229) == 1);   /* 2024 is a leap year */
    CHECK(wubu_cal_valid_date(20260230) == 0);   /* Feb 30 invalid */
    CHECK(wubu_cal_valid_date(20260131) == 1);   /* Jan 31 ok */
    CHECK(wubu_cal_valid_date(20260431) == 0);   /* Apr 31 invalid */
    CHECK(wubu_cal_valid_date(20261301) == 0);   /* month 13 */
    CHECK(wubu_cal_valid_date(20260100) == 0);   /* day 0 */
    CHECK(wubu_cal_valid_date(20260101) == 1);

    /* init */
    CHECK(wubu_cal_init(&c, 20260315) == 0);
    CHECK(c.today == 20260315);
    CHECK(c.n_events == 0);
    CHECK(wubu_cal_init(&c, 20260230) == -1);    /* invalid today */
    CHECK(wubu_cal_init(NULL, 20260315) == -1);

    /* add events */
    int e1 = wubu_cal_add(&c, 20260315, 540, 60, "Standup", WUBU_CAL_NONE);   /* 9:00 */
    CHECK(e1 == 1);
    int e2 = wubu_cal_add(&c, 20260315, 600, 30, "Review", WUBU_CAL_NONE);    /* 10:00 */
    CHECK(e2 == 2);
    int e3 = wubu_cal_add(&c, 20260316, 480, 45, "Gym", WUBU_CAL_NONE);
    CHECK(e3 == 3);
    CHECK(c.n_events == 3);

    /* bad adds */
    CHECK(wubu_cal_add(&c, 20260230, 540, 60, "Bad", WUBU_CAL_NONE) == -1);   /* invalid date */
    CHECK(wubu_cal_add(&c, 20260315, -1, 60, "Bad", WUBU_CAL_NONE) == -1);    /* start <0 */
    CHECK(wubu_cal_add(&c, 20260315, 1440, 60, "Bad", WUBU_CAL_NONE) == -1);  /* start >1439 */
    CHECK(wubu_cal_add(&c, 20260315, 540, 0, "Bad", WUBU_CAL_NONE) == -1);    /* dur 0 */
    CHECK(wubu_cal_add(&c, 20260315, 540, 60, NULL, WUBU_CAL_NONE) == -1);
    CHECK(wubu_cal_add(NULL, 20260315, 540, 60, "x", WUBU_CAL_NONE) == -1);

    /* day lookup */
    CHECK(wubu_cal_count_on(&c, 20260315) == 2);   /* standup + review */
    CHECK(wubu_cal_count_on(&c, 20260316) == 1);   /* gym */
    CHECK(wubu_cal_count_on(&c, 20260317) == 0);
    int ids[8];
    CHECK(wubu_cal_events_on(&c, 20260315, ids, 8) == 2);
    CHECK(wubu_cal_events_on(&c, 20260317, ids, 8) == 0);
    CHECK(wubu_cal_events_on(&c, 20260315, ids, 0) == 2);   /* count with max 0 */
    CHECK(wubu_cal_events_on(&c, 20260315, NULL, 8) == -1);

    /* notes */
    CHECK(wubu_cal_set_notes(&c, e1, "daily sync") == 0);
    CHECK(strcmp(c.events[0].notes, "daily sync") == 0);
    CHECK(wubu_cal_set_notes(&c, 999, "x") == -1);
    CHECK(wubu_cal_set_notes(&c, e1, NULL) == -1);

    /* reminder: standup at 540; window from 530 (8:50) of 30 min -> due */
    CHECK(wubu_cal_reminder(&c, 20260315, 530, 30) == 1);
    CHECK(wubu_cal_reminder(&c, 20260315, 500, 30) == 0);   /* 8:20, too early */
    CHECK(wubu_cal_reminder(&c, 20260315, 600, 30) == 1);   /* review at 600 due */
    CHECK(wubu_cal_reminder(&c, 20260317, 500, 60) == 0);   /* no events that day */
    CHECK(wubu_cal_reminder(&c, 20260315, -1, 30) == -1);
    CHECK(wubu_cal_reminder(&c, 20260315, 530, -1) == -1);
    CHECK(wubu_cal_reminder(NULL, 20260315, 530, 30) == -1);

    /* recurrence: daily standup from 20260315 */
    int d1 = wubu_cal_add(&c, 20260315, 540, 60, "Daily", WUBU_CAL_DAILY);
    CHECK(d1 == 4);
    CHECK(wubu_cal_count_on(&c, 20260316) == 2);   /* gym + daily */
    CHECK(wubu_cal_count_on(&c, 20260320) == 1);   /* daily only */
    CHECK(wubu_cal_count_on(&c, 20260314) == 0);   /* before anchor */

    /* weekly recurrence */
    int w1 = wubu_cal_add(&c, 20260315, 720, 60, "Weekly", WUBU_CAL_WEEKLY);
    CHECK(w1 == 5);
    CHECK(wubu_cal_count_on(&c, 20260322) == 2);   /* daily + weekly (7 days later) */
    CHECK(wubu_cal_count_on(&c, 20260316) == 2);   /* daily only (weekly not yet) */

    /* monthly recurrence (same day-of-month) */
    int m1 = wubu_cal_add(&c, 20260315, 900, 60, "Monthly", WUBU_CAL_MONTHLY);
    CHECK(m1 == 6);
    CHECK(wubu_cal_count_on(&c, 20260415) >= 2);   /* monthly + daily fall on 15th */

    /* remove */
    CHECK(wubu_cal_remove(&c, e1) == 0);   /* remove standup */
    CHECK(c.n_events == 5);
    /* on 20260315: review + daily + weekly + monthly (all anchored the 15th) */
    CHECK(wubu_cal_count_on(&c, 20260315) == 4);
    CHECK(wubu_cal_remove(&c, 999) == -1);
    CHECK(wubu_cal_remove(NULL, 1) == -1);

    if (fails == 0) printf("ALL CALENDAR TESTS PASSED\n");
    else printf("%d CALENDAR TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
