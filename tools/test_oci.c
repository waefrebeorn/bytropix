/* test_oci.c -- RUNTIME #342 tests: OCI image format + snapshot lifecycle. */
#include "wubu_oci.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_oci_t o;

    CHECK(wubu_oci_init(&o) == 0);

    /* blob store */
    CHECK(wubu_oci_put_blob(&o, "sha256:aaaa", 100, "application/vnd.oci.image.layer.v1.tar") == 0);
    CHECK(wubu_oci_put_blob(&o, "sha256:bbbb", 200, "application/vnd.oci.image.config.v1+json") == 0);
    CHECK(wubu_oci_put_blob(&o, "sha256:cccc", 50, "application/vnd.oci.image.manifest.v1+json") == 0);
    CHECK(wubu_oci_put_blob(&o, "sha256:aaaa", 100, "x") == 0);   /* dup: idempotent */
    CHECK(wubu_oci_has_blob(&o, "sha256:aaaa") == 1);
    CHECK(wubu_oci_has_blob(&o, "sha256:zzzz") == 0);
    CHECK(wubu_oci_blob_size(&o, "sha256:bbbb") == 200);
    CHECK(wubu_oci_blob_size(&o, "sha256:zzzz") == -1);

    /* image: manifest cccc, config bbbb, layers [aaaa] */
    const char *layers[] = { "sha256:aaaa" };
    CHECK(wubu_oci_set_image(&o, "myimg", "sha256:cccc", "sha256:bbbb", layers, 1) == 0);
    CHECK(wubu_oci_image_complete(&o) == 1);

    /* image referencing a missing blob fails */
    const char *badlayers[] = { "sha256:dddd" };
    CHECK(wubu_oci_set_image(&o, "bad", "sha256:cccc", "sha256:bbbb", badlayers, 1) == -1);
    /* original image still intact */
    CHECK(wubu_oci_image_complete(&o) == 1);

    /* snapshot lifecycle: base (no parent) */
    CHECK(wubu_oci_prepare(&o, "base", "") == 0);
    CHECK(wubu_oci_snap_state(&o, "base") == WUBU_OCI_SNAP_ACTIVE);
    CHECK(wubu_oci_prepare(&o, "base", "") == -1);   /* dup */

    /* child needs a COMMITTED parent */
    CHECK(wubu_oci_prepare(&o, "child", "base") == -1);   /* base still active */
    CHECK(wubu_oci_commit(&o, "base") == 0);
    CHECK(wubu_oci_snap_state(&o, "base") == WUBU_OCI_SNAP_COMMITTED);
    CHECK(wubu_oci_commit(&o, "base") == -1);             /* already committed */

    /* now prepare a child of committed base */
    CHECK(wubu_oci_prepare(&o, "child", "base") == 0);
    CHECK(wubu_oci_snap_state(&o, "child") == WUBU_OCI_SNAP_ACTIVE);

    /* mount/unmount */
    CHECK(wubu_oci_mount(&o, "child") == 0);
    CHECK(wubu_oci_mount(&o, "child") == -1);      /* already mounted */
    CHECK(wubu_oci_unmount(&o, "child") == 0);
    CHECK(wubu_oci_unmount(&o, "child") == -1);    /* not mounted */

    /* cannot remove base while child references it */
    CHECK(wubu_oci_remove(&o, "base") == -1);      /* has child */
    /* cannot remove a mounted snapshot */
    CHECK(wubu_oci_mount(&o, "child") == 0);
    CHECK(wubu_oci_remove(&o, "child") == -1);     /* mounted */
    CHECK(wubu_oci_unmount(&o, "child") == 0);

    /* remove child (unmounted, no children), then base */
    CHECK(wubu_oci_remove(&o, "child") == 0);
    CHECK(wubu_oci_snap_state(&o, "child") == -1);   /* gone */
    CHECK(wubu_oci_remove(&o, "base") == 0);
    CHECK(wubu_oci_snap_state(&o, "base") == -1);

    /* prepare with a nonexistent parent fails */
    CHECK(wubu_oci_prepare(&o, "orphan", "ghost") == -1);

    /* bad args */
    CHECK(wubu_oci_mount(&o, "nope") == -1);
    CHECK(wubu_oci_commit(&o, "nope") == -1);
    CHECK(wubu_oci_remove(&o, "nope") == -1);
    CHECK(wubu_oci_snap_state(&o, "nope") == -1);
    CHECK(wubu_oci_init(NULL) == -1);
    CHECK(wubu_oci_image_complete(&o) == 1);   /* image still complete after snap churn */

    if (fails == 0) printf("ALL OCI TESTS PASSED\n");
    else printf("%d OCI TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
