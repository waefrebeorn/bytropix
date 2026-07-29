#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "wubu_kv_styx.h"

int main(void) {
    int rc = 0;

    wubu_kv_styx_init();

    /* empty snapshot */
    char *snap = wubu_kv_styx_snapshot_json(NULL);
    assert(snap != NULL);
    assert(strstr(snap, "\"registered\":0") != NULL);
    free(snap);

    /* register two layers */
    static char layer0[4096];
    static char layer1[2048];
    assert(wubu_kv_styx_register("/n/kv/0", layer0, sizeof(layer0)) == 0);
    assert(wubu_kv_styx_register("/n/kv/1", layer1, sizeof(layer1)) == 0);
    assert(wubu_kv_styx_registered_count() == 2);

    /* lookup */
    size_t out_bytes = 0;
    assert(wubu_kv_styx_lookup("/n/kv/0", &out_bytes) == layer0);
    assert(out_bytes == sizeof(layer0));
    assert(wubu_kv_styx_lookup("/n/kv/1", NULL) == layer1);

    /* unregister */
    assert(wubu_kv_styx_unregister("/n/kv/0") == 0);
    assert(wubu_kv_styx_registered_count() == 1);
    assert(wubu_kv_styx_unregister("/n/kv/99") == -1);

    /* snapshot with one entry */
    snap = wubu_kv_styx_snapshot_json(NULL);
    assert(snap != NULL);
    assert(strstr(snap, "\"registered\":1") != NULL);
    assert(strstr(snap, "\"path\":\"/n/kv/1\"") != NULL);
    free(snap);

    wubu_kv_styx_shutdown();
    printf("PASS: wubu_kv_styx public API\n");
    return rc;
}
