/* wubu_bvh.c -- WORLDSIM #269 BVH acceleration structure.
 * See wubu_bvh.h. Top-down median split, fixed arena, deterministic. */
#include "wubu_bvh.h"

int wubu_bvh_init(wubu_bvh_t *t) {
    if (!t) return -1;
    t->n_nodes = 0;
    t->n_bodies = 0;
    return 0;
}

int wubu_bvh_add(wubu_bvh_t *t, double x, double y, double r, int id) {
    if (!t || t->n_bodies >= WUBU_BVH_MAX_BODIES) return -1;
    wubu_bvh_body_t *b = &t->bodies[t->n_bodies];
    b->x = x; b->y = y; b->r = r; b->id = id;
    return t->n_bodies++;
}

static int new_node(wubu_bvh_t *t) {
    if (t->n_nodes >= WUBU_BVH_MAX_NODES) return -1;
    int i = t->n_nodes++;
    t->nodes[i].left = t->nodes[i].right = -1;
    t->nodes[i].body = -1;
    return i;
}

/* indices[] is a scratch array of body indices for this subtree. */
static int build_rec(wubu_bvh_t *t, int *idx, int lo, int hi) {
    int node = new_node(t);
    if (node < 0) return -1;
    /* compute AABB over idx[lo..hi) */
    double minx = 1e18, miny = 1e18, maxx = -1e18, maxy = -1e18;
    for (int i = lo; i < hi; i++) {
        wubu_bvh_body_t *b = &t->bodies[idx[i]];
        if (b->x - b->r < minx) minx = b->x - b->r;
        if (b->y - b->r < miny) miny = b->y - b->r;
        if (b->x + b->r > maxx) maxx = b->x + b->r;
        if (b->y + b->r > maxy) maxy = b->y + b->r;
    }
    t->nodes[node].minx = minx; t->nodes[node].miny = miny;
    t->nodes[node].maxx = maxx; t->nodes[node].maxy = maxy;

    if (hi - lo <= 1) {
        t->nodes[node].body = idx[lo];
        return node;
    }
    /* median split on the wider axis */
    int mid = (lo + hi) / 2;
    double dx = maxx - minx, dy = maxy - miny;
    /* simple insertion sort of idx[lo..hi) by the split coordinate */
    int axis = (dx >= dy) ? 0 : 1;
    for (int i = lo + 1; i < hi; i++) {
        int key = idx[i];
        double kv = axis == 0 ? t->bodies[key].x : t->bodies[key].y;
        int j = i - 1;
        while (j >= lo) {
            double jv = axis == 0 ? t->bodies[idx[j]].x : t->bodies[idx[j]].y;
            if (jv <= kv) break;
            idx[j + 1] = idx[j];
            j--;
        }
        idx[j + 1] = key;
    }
    int l = build_rec(t, idx, lo, mid);
    int r = build_rec(t, idx, mid, hi);
    t->nodes[node].left = l;
    t->nodes[node].right = r;
    return node;
}

int wubu_bvh_build(wubu_bvh_t *t) {
    if (!t || t->n_bodies == 0) return -1;
    int idx[WUBU_BVH_MAX_BODIES];
    for (int i = 0; i < t->n_bodies; i++) idx[i] = i;
    int root = build_rec(t, idx, 0, t->n_bodies);
    return root < 0 ? -1 : t->n_nodes;
}

static int query_rec(const wubu_bvh_t *t, int node, double qx0, double qy0, double qx1, double qy1) {
    if (node < 0) return 0;
    const wubu_bvh_node_t *n = &t->nodes[node];
    /* AABB overlap test (prune if disjoint) */
    if (n->maxx < qx0 || n->minx > qx1 || n->maxy < qy0 || n->miny > qy1) return 0;
    if (n->body >= 0) return 1;   /* leaf overlaps */
    return query_rec(t, n->left, qx0, qy0, qx1, qy1) +
           query_rec(t, n->right, qx0, qy0, qx1, qy1);
}

int wubu_bvh_query(const wubu_bvh_t *t, double qx0, double qy0, double qx1, double qy1) {
    if (!t || t->n_nodes == 0) return 0;
    return query_rec(t, 0, qx0, qy0, qx1, qy1);
}

static int depth_rec(const wubu_bvh_t *t, int node) {
    if (node < 0) return 0;
    const wubu_bvh_node_t *n = &t->nodes[node];
    if (n->body >= 0) return 1;
    int dl = depth_rec(t, n->left), dr = depth_rec(t, n->right);
    return 1 + (dl > dr ? dl : dr);
}

int wubu_bvh_depth(const wubu_bvh_t *t) {
    if (!t || t->n_nodes == 0) return 0;
    return depth_rec(t, 0);
}
