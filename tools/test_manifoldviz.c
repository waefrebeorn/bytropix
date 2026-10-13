/* test_manifoldviz.c -- BRAIN #127 manifold visualization.
 * Proves: origin -> center, perspective scaling, boundary depth, clamp inside
 * ball, batch projection, fail-closed on NULL. */
#include "wubu_manifoldviz.h"
#include <stdio.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_mv_t r;
    CHECK(wubu_mv_init(&r, 2.0) == 0, "init ok (focal 2.0)");
    CHECK(wubu_mv_init(&r, 0) == -1, "zero focal -> -1");

    /* origin -> center, depth 0 */
    double p0[WUBU_MV_DIM] = {0,0,0, 0,0,0,0,0};
    wubu_mv_pixel_t px;
    CHECK(wubu_mv_project(&r, p0, &px) == 0, "project origin ok");
    CHECK(fabs(px.x) < 1e-9 && fabs(px.y) < 1e-9, "origin -> screen center (0,0)");
    CHECK(fabs(px.depth) < 1e-9, "origin depth 0");
    CHECK(px.visible == 1, "origin visible");

    /* a point off-center projects off-center */
    double p1[WUBU_MV_DIM] = {0.3, 0.1, 0, 0,0,0,0,0};
    wubu_mv_project(&r, p1, &px);
    CHECK(px.x > 0 && px.y > 0, "positive point -> positive screen coords");
    CHECK(px.depth > 0.3, "depth reflects distance from center");

    /* a point near the boundary has depth near 1 */
    double p2[WUBU_MV_DIM] = {0.95, 0, 0, 0,0,0,0,0};
    wubu_mv_project(&r, p2, &px);
    CHECK(px.depth > 0.9, "boundary point depth near 1");

    /* a point outside the ball is clamped inside (still visible, depth < 1) */
    double p3[WUBU_MV_DIM] = {5.0, 0, 0, 0,0,0,0,0};
    wubu_mv_project(&r, p3, &px);
    CHECK(px.depth < 1.0, "out-of-ball point clamped inside");
    CHECK(px.visible == 1, "clamped point visible");

    /* perspective: a point closer to viewer (+z) projects larger than the same
     * x,y at z=0 */
    double pz0[WUBU_MV_DIM] = {0.2, 0, 0, 0,0,0,0,0};
    double pz1[WUBU_MV_DIM] = {0.2, 0, 0.5, 0,0,0,0,0};
    wubu_mv_pixel_t a, b;
    wubu_mv_project(&r, pz0, &a);
    wubu_mv_project(&r, pz1, &b);
    CHECK(b.x > a.x, "nearer point (+z) projects larger (perspective)");

    /* batch projection */
    double pts[2 * WUBU_MV_DIM] = {0,0,0,0,0,0,0,0,  0.1,0.1,0,0,0,0,0,0};
    wubu_mv_pixel_t out[2];
    CHECK(wubu_mv_project_all(&r, pts, 2, out) == 2, "batch project 2 points");
    CHECK(fabs(out[0].x) < 1e-9 && out[1].x > 0, "batch: first center, second off-center");

    /* fail-closed */
    CHECK(wubu_mv_project(&r, p0, NULL) == -1, "NULL out -> -1");
    CHECK(wubu_mv_project(NULL, p0, &px) == -1, "NULL renderer -> -1");
    CHECK(wubu_mv_project_all(&r, pts, 0, out) == -1, "zero count -> -1");
    CHECK(wubu_mv_project_all(NULL, pts, 2, out) == -1, "NULL batch -> -1");
    CHECK(wubu_mv_init(NULL, 2.0) == -1, "NULL init -> -1");

    printf("\n=== test_manifoldviz: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
