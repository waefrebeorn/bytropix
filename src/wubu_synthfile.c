/* wubu_synthfile.c -- NAMESPACE #195 9P synthetic files (read computes).
 * See wubu_synthfile.h. */
#include "wubu_synthfile.h"
#include <string.h>

int wubu_synthfs_init(wubu_synthfs_t *fs) {
    if (!fs) return -1;
    memset(fs->files, 0, sizeof fs->files);
    fs->n = 0;
    return 0;
}

int wubu_synthfs_add(wubu_synthfs_t *fs, const char *name,
                     wubu_synth_gen_t gen, void *ctx) {
    if (!fs || !name || !gen || fs->n >= 16) return -1;
    if (strlen(name) >= 64) return -1;
    strncpy(fs->files[fs->n].name, name, 63);
    fs->files[fs->n].name[63] = '\0';
    fs->files[fs->n].gen = gen;
    fs->files[fs->n].ctx = ctx;
    fs->n++;
    return 0;
}

int wubu_synthfs_read(const wubu_synthfs_t *fs, const char *name,
                      char *buf, size_t cap) {
    if (!fs || !name || !buf || cap == 0) return -1;
    for (int i = 0; i < fs->n; i++) {
        if (strcmp(fs->files[i].name, name) == 0) {
            if (!fs->files[i].gen) return -1;
            /* invoke the generator: it computes the content on demand */
            return fs->files[i].gen(fs->files[i].ctx, buf, cap);
        }
    }
    return -1;   /* unknown file */
}
