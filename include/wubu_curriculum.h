#ifndef WUBU_CURRICULUM_H
#define WUBU_CURRICULUM_H
/* wubu_curriculum.h — curriculum learning schedule (BRAIN #242).
 *
 * A competence-based curriculum ramps the difficulty threshold from `d0` (easy)
 * at progress t=0 to `d1` (hard) at t=1, linearly. A sample of difficulty `s`
 * is eligible when s <= threshold(t). Early training sees only easy samples;
 * later training admits harder ones. The forward (wubu_curriculum_threshold +
 * wubu_curriculum_eligible) is covered by test_curriculum with measured
 * assertions.
 */
#include <stddef.h>

/* The difficulty threshold at training progress t (clamped to [0,1]):
 * d0 + (d1 - d0) * t. */
double wubu_curriculum_threshold(double d0, double d1, double t);

/* Is a sample of difficulty `s` eligible at progress `t` under the schedule?
 * Returns 1 if s <= threshold(t), else 0. */
int wubu_curriculum_eligible(double d0, double d1, double t, double s);

#endif
