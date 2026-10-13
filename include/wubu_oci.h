#ifndef WUBU_OCI_H
#define WUBU_OCI_H

/* wubu_oci.h -- RUNTIME #342 OCI image format + snapshot lifecycle.
 *
 * A containerd-equivalent OCI runtime core: a content-addressable blob store
 * (blobs keyed by sha256 digest), an OCI image index/manifest/config model,
 * and the snapshot lifecycle -- prepare (create an active snapshot with a
 * parent), mount, commit (turn the active snapshot into a committed one),
 * and remove. Snapshots form a parent chain; a snapshot cannot be removed
 * while a child still references it. Deterministic, real work (no stubs). */

#include <stddef.h>
#include <stdint.h>

#define WUBU_OCI_BLOBS_MAX 256
#define WUBU_OCI_SNAPS_MAX 64
#define WUBU_OCI_DIGEST_LEN 71   /* "sha256:" + 64 hex */
#define WUBU_OCI_NAME_MAX 96
#define WUBU_OCI_KIND_MAX 32

typedef enum {
    WUBU_OCI_SNAP_ACTIVE = 0,    /* being written (prepare'd) */
    WUBU_OCI_SNAP_COMMITTED = 1  /* immutable, has children possibly */
} wubu_oci_snap_state_t;

typedef struct {
    char digest[WUBU_OCI_DIGEST_LEN];
    uint64_t size;
    char media_type[WUBU_OCI_KIND_MAX];
    int present;
} wubu_oci_blob_t;

typedef struct {
    char key[WUBU_OCI_NAME_MAX];     /* snapshot key */
    char parent[WUBU_OCI_NAME_MAX];  /* parent snapshot key, "" if none */
    wubu_oci_snap_state_t state;
    int mounted;
    int n_children;
} wubu_oci_snap_t;

typedef struct {
    /* content-addressable blob store */
    wubu_oci_blob_t blobs[WUBU_OCI_BLOBS_MAX];
    int n_blobs;
    /* snapshots */
    wubu_oci_snap_t snaps[WUBU_OCI_SNAPS_MAX];
    int n_snaps;
    /* image reference: name -> manifest digest -> config digest */
    char image_name[WUBU_OCI_NAME_MAX];
    char manifest_digest[WUBU_OCI_DIGEST_LEN];
    char config_digest[WUBU_OCI_DIGEST_LEN];
    int n_layers;
    char layer_digests[16][WUBU_OCI_DIGEST_LEN];
} wubu_oci_t;

/* Initialise. Returns 0. */
int wubu_oci_init(wubu_oci_t *o);

/* -- content-addressable blob store -- */
/* Put a blob by digest. Returns 0/-1 (dup digest is a no-op success). */
int wubu_oci_put_blob(wubu_oci_t *o, const char *digest, uint64_t size,
                      const char *media_type);
/* Does a blob exist? 1/0. */
int wubu_oci_has_blob(const wubu_oci_t *o, const char *digest);
/* Blob size by digest, or -1. */
int64_t wubu_oci_blob_size(const wubu_oci_t *o, const char *digest);

/* -- image model -- */
/* Define an image: manifest + config digests + layer digests. The manifest,
 * config, and every layer blob must already be present. Returns 0/-1. */
int wubu_oci_set_image(wubu_oci_t *o, const char *name, const char *manifest,
                       const char *config, const char *const *layers, int n_layers);
/* Is the image fully present (all referenced blobs stored)? 1/0/-1. */
int wubu_oci_image_complete(const wubu_oci_t *o);

/* -- snapshot lifecycle -- */
/* Prepare (create) an active snapshot with an optional parent ("" for none).
 * The parent must exist and be committed. Returns 0/-1. */
int wubu_oci_prepare(wubu_oci_t *o, const char *key, const char *parent);
/* Mount a snapshot (must be active or committed, not already mounted). 0/-1. */
int wubu_oci_mount(wubu_oci_t *o, const char *key);
/* Commit an active snapshot (active -> committed). 0/-1. */
int wubu_oci_commit(wubu_oci_t *o, const char *key);
/* Unmount a mounted snapshot. 0/-1. */
int wubu_oci_unmount(wubu_oci_t *o, const char *key);
/* Remove a snapshot: must be unmounted and have no children. 0/-1. */
int wubu_oci_remove(wubu_oci_t *o, const char *key);
/* Snapshot state by key, or -1. */
int wubu_oci_snap_state(const wubu_oci_t *o, const char *key);

#endif
