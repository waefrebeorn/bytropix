/* wubu_companion.c -- UX #246 Bonzi companion AGI-UX persona.
 * See wubu_companion.h. */
#include "wubu_companion.h"

int wubu_comp_init(wubu_companion_t *c) {
    if (!c) return -1;
    c->mood = WUBU_COMP_IDLE;
    c->n_transitions = 0;
    c->ticks_in_mood = 0;
    return 0;
}

int wubu_comp_set(wubu_companion_t *c, wubu_comp_mood_t mood) {
    if (!c) return -1;
    if (mood < WUBU_COMP_IDLE || mood > WUBU_COMP_CONCERN) return -1;
    if (mood != c->mood) {
        c->mood = mood;
        c->n_transitions++;
        c->ticks_in_mood = 0;
    }
    return 0;
}

int wubu_comp_tick(wubu_companion_t *c) {
    if (!c) return -1;
    c->ticks_in_mood++;
    return 0;
}

wubu_comp_mood_t wubu_comp_mood(const wubu_companion_t *c) {
    return c ? c->mood : WUBU_COMP_IDLE;
}

const char *wubu_comp_expression(const wubu_companion_t *c) {
    if (!c) return "";
    switch (c->mood) {
        case WUBU_COMP_IDLE:     return "Ready when you are.";
        case WUBU_COMP_WORKING:  return "On it...";
        case WUBU_COMP_ALERT:    return "Heads up -- something needs you.";
        case WUBU_COMP_CELEBRATE:return "Goal complete! Nice.";
        case WUBU_COMP_CONCERN:  return "Hit a snag -- investigating.";
        default:                 return "";
    }
}

int wubu_comp_transitions(const wubu_companion_t *c) {
    return c ? c->n_transitions : 0;
}
