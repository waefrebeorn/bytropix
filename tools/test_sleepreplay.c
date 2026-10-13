/* test_sleepreplay.c -- BRAIN #125 memory consolidation sleep-replay.
 * Proves: replay strengthens, no-replay decays, consolidation threshold,
 * replay resists forgetting, fail-closed on NULL. */
#include "wubu_sleepreplay.h"
#include <stdio.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_sleepreplay_t s;
    /* decay 0.1/sleep, replay_gain 0.3, threshold 0.8 */
    CHECK(wubu_sr_init(&s, 0.1, 0.3) == 0, "init ok (decay 0.1, gain 0.3)");

    /* record three episodes at strength 0.5 */
    wubu_sr_record(&s, 1);
    wubu_sr_record(&s, 2);
    wubu_sr_record(&s, 3);
    CHECK(wubu_sr_strength(&s, 1) == 0.5, "fresh episode strength 0.5");

    /* sleep replaying only episode 1: 0.5 - 0.1 + 0.3 = 0.7 (not yet consolidated) */
    int replay1[] = {1};
    int cons = wubu_sr_sleep(&s, replay1, 1);
    CHECK(wubu_sr_strength(&s, 1) > 0.69 && wubu_sr_strength(&s, 1) < 0.71, "replayed ep1 -> 0.7");
    CHECK(wubu_sr_strength(&s, 2) < 0.41, "unreplayed ep2 decayed -> 0.4");
    CHECK(cons == 0, "nothing consolidated yet (0.7 < 0.8)");

    /* sleep again replaying ep1: 0.7 - 0.1 + 0.3 = 0.9 -> consolidated */
    cons = wubu_sr_sleep(&s, replay1, 1);
    CHECK(wubu_sr_strength(&s, 1) > 0.89 && wubu_sr_strength(&s, 1) < 0.91, "ep1 -> 0.9");
    CHECK(wubu_sr_consolidated(&s, 1) == 1, "ep1 consolidated (>= 0.8)");
    CHECK(cons == 1, "1 consolidated");

    /* ep2 keeps decaying (unreplayed): 0.4 -> 0.3 -> 0.2 ... */
    wubu_sr_sleep(&s, replay1, 1);
    wubu_sr_sleep(&s, replay1, 1);
    CHECK(wubu_sr_strength(&s, 2) < 0.21, "unreplayed ep2 keeps decaying toward 0");
    CHECK(wubu_sr_consolidated(&s, 2) == 0, "ep2 never consolidated (forgotten)");

    /* strength capped at 1 */
    for (int i = 0; i < 10; i++) wubu_sr_sleep(&s, replay1, 1);
    CHECK(wubu_sr_strength(&s, 1) <= 1.0, "strength capped at 1.0");

    /* replay count tracked */
    CHECK(s.eps[0].replay_count >= 12, "ep1 replayed many times");

    /* fail-closed */
    CHECK(wubu_sr_init(NULL, 0.1, 0.3) == -1, "NULL init -> -1");
    CHECK(wubu_sr_record(NULL, 1) == -1, "NULL record -> -1");
    CHECK(wubu_sr_sleep(NULL, replay1, 1) == -1, "NULL sleep -> -1");
    CHECK(wubu_sr_strength(&s, 999) == 0, "unknown id strength -> 0");
    CHECK(wubu_sr_consolidated(&s, 999) == 0, "unknown id consolidated -> 0");
    CHECK(wubu_sr_n_consolidated(NULL) == 0, "NULL n_consolidated -> 0");

    printf("\n=== test_sleepreplay: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
