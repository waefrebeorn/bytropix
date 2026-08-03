/* wubu_traj_sft.c -- the trajectory -> masked-observation SFT converter.
 * The input is COPIED (never modified in place -- the in-place NUL
 * termination clobbered the '\n' separators and broke re-parsing). */
#include <stdlib.h>
#include <string.h>
#include "wubu_traj_sft.h"

int wubu_traj_sft_convert(const char *traj, wubu_sft_seg_t *segs,
                          int max_segs)
{
    if (!traj || !segs || max_segs < 1) return 0;
    size_t tlen = strlen(traj);
    char *copy = (char *)malloc(tlen + 1);
    if (!copy) return 0;
    memcpy(copy, traj, tlen + 1);

    int n = 0;
    char *p = copy;
    while (*p && n < max_segs) {
        char *eol = strchr(p, '\n');
        size_t len = eol ? (size_t)(eol - p) : strlen(p);
        char *line = p;
        while (len > 0 && line[len - 1] == '\r') len--;
        if (len > 0) {
            int type = -1;
            char *body = NULL;
            const char *tags[] = { "[user]", "[obs]", "[tool_resp]",
                                   "[assistant]", "[think]", "[tool_call]" };
            for (int i = 0; i < 6; i++) {
                size_t tl = strlen(tags[i]);
                if (len > tl && strncmp(line, tags[i], tl) == 0 &&
                    line[tl] == ' ') {
                    type = i;
                    body = line + tl + 1;
                    len = len - tl - 1;   /* the body length */
                    break;
                }
            }
            if (type >= 0) {
                body[len] = '\0';         /* safe: inside the copy */
                segs[n].type = type;
                segs[n].text = body;
                segs[n].train = (type == WUBU_SEG_ASSISTANT ||
                                 type == WUBU_SEG_THINK ||
                                 type == WUBU_SEG_TOOL_CALL) ? 1 : 0;
                n++;
            }
        }
        if (!eol) break;
        p = eol + 1;
    }
    for (int i = 0; i < n; i++) segs[i].base = copy;
    if (n == 0) { free(copy); return 0; }
    return n;
}

void wubu_traj_sft_segs_free(wubu_sft_seg_t *segs, int n)
{
    if (n > 0 && segs && segs[0].base) free((void *)segs[0].base);
}

float wubu_traj_sft_train_frac(const wubu_sft_seg_t *segs, int n)
{
    if (!segs || n < 1) return 0;
    int tr = 0;
    for (int i = 0; i < n; i++) if (segs[i].train) tr++;
    return (float)tr / (float)n;
}
