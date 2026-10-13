/* wubu_oci.c -- RUNTIME #342 OCI image format + snapshot lifecycle.
 * See wubu_oci.h. Real, deterministic containerd-equivalent core. */

#include "wubu_oci.h"
#include <string.h>

int wubu_oci_init(wubu_oci_t *o)
{
    if (!o) return -1;
    memset(o, 0, sizeof(*o));
    return 0;
}

/* -- blob store -- */
static wubu_oci_blob_t *find_blob(wubu_oci_t *o, const char *digest)
{
    for (int i = 0; i < o->n_blobs; i++)
        if (o->blobs[i].present && strcmp(o->blobs[i].digest, digest) == 0)
            return &o->blobs[i];
    return NULL;
}

int wubu_oci_put_blob(wubu_oci_t *o, const char *digest, uint64_t size,
                      const char *media_type)
{
    if (!o || !digest) return -1;
    if (find_blob(o, digest)) return 0;   /* dup: idempotent */
    if (o->n_blobs >= WUBU_OCI_BLOBS_MAX) return -1;
    wubu_oci_blob_t *b = &o->blobs[o->n_blobs++];
    memset(b, 0, sizeof(*b));
    strncpy(b->digest, digest, WUBU_OCI_DIGEST_LEN - 1);
    b->size = size;
    if (media_type) strncpy(b->media_type, media_type, WUBU_OCI_KIND_MAX - 1);
    b->present = 1;
    return 0;
}

int wubu_oci_has_blob(const wubu_oci_t *o, const char *digest)
{
    if (!o || !digest) return 0;
    for (int i = 0; i < o->n_blobs; i++)
        if (o->blobs[i].present && strcmp(o->blobs[i].digest, digest) == 0) return 1;
    return 0;
}

int64_t wubu_oci_blob_size(const wubu_oci_t *o, const char *digest)
{
    if (!o || !digest) return -1;
    for (int i = 0; i < o->n_blobs; i++)
        if (o->blobs[i].present && strcmp(o->blobs[i].digest, digest) == 0)
            return (int64_t)o->blobs[i].size;
    return -1;
}

/* -- image model -- */
int wubu_oci_set_image(wubu_oci_t *o, const char *name, const char *manifest,
                       const char *config, const char *const *layers, int n_layers)
{
    if (!o || !name || !manifest || !config) return -1;
    if (n_layers < 0 || n_layers > 16) return -1;
    /* all referenced blobs must be present */
    if (!wubu_oci_has_blob(o, manifest)) return -1;
    if (!wubu_oci_has_blob(o, config)) return -1;
    for (int i = 0; i < n_layers; i++)
        if (!layers[i] || !wubu_oci_has_blob(o, layers[i])) return -1;
    strncpy(o->image_name, name, WUBU_OCI_NAME_MAX - 1);
    strncpy(o->manifest_digest, manifest, WUBU_OCI_DIGEST_LEN - 1);
    strncpy(o->config_digest, config, WUBU_OCI_DIGEST_LEN - 1);
    o->n_layers = n_layers;
    for (int i = 0; i < n_layers; i++)
        strncpy(o->layer_digests[i], layers[i], WUBU_OCI_DIGEST_LEN - 1);
    return 0;
}

int wubu_oci_image_complete(const wubu_oci_t *o)
{
    if (!o || o->image_name[0] == '\0') return -1;
    if (!wubu_oci_has_blob(o, o->manifest_digest)) return 0;
    if (!wubu_oci_has_blob(o, o->config_digest)) return 0;
    for (int i = 0; i < o->n_layers; i++)
        if (!wubu_oci_has_blob(o, o->layer_digests[i])) return 0;
    return 1;
}

/* -- snapshots -- */
static wubu_oci_snap_t *find_snap(wubu_oci_t *o, const char *key)
{
    for (int i = 0; i < o->n_snaps; i++)
        if (strcmp(o->snaps[i].key, key) == 0) return &o->snaps[i];
    return NULL;
}

int wubu_oci_prepare(wubu_oci_t *o, const char *key, const char *parent)
{
    if (!o || !key) return -1;
    if (find_snap(o, key)) return -1;   /* dup */
    if (o->n_snaps >= WUBU_OCI_SNAPS_MAX) return -1;
    if (parent && parent[0]) {
        wubu_oci_snap_t *p = find_snap(o, parent);
        if (!p) return -1;                        /* parent must exist */
        if (p->state != WUBU_OCI_SNAP_COMMITTED) return -1;  /* parent committed */
        if (p->n_children >= 0) p->n_children++;
    }
    wubu_oci_snap_t *s = &o->snaps[o->n_snaps++];
    memset(s, 0, sizeof(*s));
    strncpy(s->key, key, WUBU_OCI_NAME_MAX - 1);
    if (parent && parent[0]) strncpy(s->parent, parent, WUBU_OCI_NAME_MAX - 1);
    s->state = WUBU_OCI_SNAP_ACTIVE;
    s->mounted = 0;
    return 0;
}

int wubu_oci_mount(wubu_oci_t *o, const char *key)
{
    if (!o) return -1;
    wubu_oci_snap_t *s = find_snap(o, key);
    if (!s) return -1;
    if (s->mounted) return -1;   /* already mounted */
    s->mounted = 1;
    return 0;
}

int wubu_oci_commit(wubu_oci_t *o, const char *key)
{
    if (!o) return -1;
    wubu_oci_snap_t *s = find_snap(o, key);
    if (!s) return -1;
    if (s->state != WUBU_OCI_SNAP_ACTIVE) return -1;  /* only active -> committed */
    s->state = WUBU_OCI_SNAP_COMMITTED;
    return 0;
}

int wubu_oci_unmount(wubu_oci_t *o, const char *key)
{
    if (!o) return -1;
    wubu_oci_snap_t *s = find_snap(o, key);
    if (!s) return -1;
    if (!s->mounted) return -1;   /* not mounted */
    s->mounted = 0;
    return 0;
}

int wubu_oci_remove(wubu_oci_t *o, const char *key)
{
    if (!o) return -1;
    wubu_oci_snap_t *s = find_snap(o, key);
    if (!s) return -1;
    if (s->mounted) return -1;         /* must be unmounted */
    if (s->n_children > 0) return -1;  /* children still reference it */
    /* decrement parent's child count */
    if (s->parent[0]) {
        wubu_oci_snap_t *p = find_snap(o, s->parent);
        if (p && p->n_children > 0) p->n_children--;
    }
    s->key[0] = '\0';   /* mark removed (slot reusable via key check) */
    return 0;
}

int wubu_oci_snap_state(const wubu_oci_t *o, const char *key)
{
    if (!o) return -1;
    for (int i = 0; i < o->n_snaps; i++)
        if (o->snaps[i].key[0] && strcmp(o->snaps[i].key, key) == 0)
            return (int)o->snaps[i].state;
    return -1;
}
