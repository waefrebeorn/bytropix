/* wubu_curriculum.c — curriculum learning schedule (BRAIN #242).
 * C11, self-contained. Linear difficulty ramp d0->d1 over progress t in [0,1].
 */
#include "wubu_curriculum.h"

double wubu_curriculum_threshold(double d0, double d1, double t) {
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    return d0 + (d1 - d0) * t;
}

int wubu_curriculum_eligible(double d0, double d1, double t, double s) {
    /* Small epsilon so a sample exactly at the (float-rounded) threshold is
     * admitted; avoids 0.9 <= 0.8999999999999999 boundary misses. */
    return s <= wubu_curriculum_threshold(d0, d1, t) + 1e-9 ? 1 : 0;
}
