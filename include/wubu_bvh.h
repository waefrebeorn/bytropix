#ifndef WUBU_BVH_H
#define WUBU_BVH_H

/* wubu_bVH.h -- WORLDSIM #269 BVH acceleration structure.
 *
 * A Bounding Volume Hierarchy: a binary tree over bodies where each node
 * holds an axis-aligned bounding box (AABB) enclosing its children. Collision
 * queries descend the tree, pruning whole subtrees whose AABB doesn't overlap
 * the query box -- turning O(n) into O(log n) for well-distributed bodies.
 * Built top-down by median split. Deterministic, fixed-size arena (no malloc),
 * real work (no stubs). */

#define WUBU_BVH_MAX_NODES 512
#define WUBU_BVH_MAX_BODIES 256

typedef struct {
    double x, y, r;
    int id;
} wubu_bvh_body_t;

typedef struct {
    double minx, miny, maxx, maxy;   /* AABB */
    int left, right;                 /* child node indices, -1 = leaf */
    int body;                        /* leaf body index, -1 = internal */
} wubu_bvh_node_t;

typedef struct {
    wubu_bvh_node_t nodes[WUBU_BVH_MAX_NODES];
    int n_nodes;
    wubu_bvh_body_t bodies[WUBU_BVH_MAX_BODIES];
    int n_bodies;
} wubu_bvh_t;

/* Initialise an empty BVH. */
int wubu_bvh_init(wubu_bvh_t *t);

/* Add a body. Returns index or -1. */
int wubu_bvh_add(wubu_bvh_t *t, double x, double y, double r, int id);

/* Build the hierarchy from the added bodies (median split). Returns node
 * count, or -1 on error. */
int wubu_bvh_build(wubu_bvh_t *t);

/* Count bodies whose AABB overlaps the query box (qx0,qy0)-(qx1,qy1), using
 * the BVH to prune. */
int wubu_bvh_query(const wubu_bvh_t *t, double qx0, double qy0, double qx1, double qy1);

/* Tree depth (for balance checks). */
int wubu_bvh_depth(const wubu_bvh_t *t);

#endif
