/* wubu_ftrace.c -- KERNEL #30 ftrace ring buffer as /n/kernel/ftrace.
 * See wubu_ftrace.h. Real, deterministic ring buffer. */

#include "wubu_ftrace.h"
#include <string.h>
#include <stdio.h>

int wubu_ftrace_init(wubu_ftrace_t *f)
{
    if (!f) return -1;
    memset(f, 0, sizeof(*f));
    return 0;
}

int wubu_ftrace_set_enabled(wubu_ftrace_t *f, int enabled)
{
    if (!f) return -1;
    f->enabled = enabled ? 1 : 0;
    return 0;
}

int wubu_ftrace_clear(wubu_ftrace_t *f)
{
    if (!f) return -1;
    f->head = 0;
    f->count = 0;
    f->dropped = 0;
    f->total = 0;
    f->cur_depth = 0;
    return 0;
}

static void record(wubu_ftrace_t *f, const char *func, uint64_t ts, int is_exit, int depth)
{
    f->total++;
    if (f->count == WUBU_FTRACE_CAP) f->dropped++;   /* overwriting oldest */
    wubu_ftrace_rec_t *r = &f->ring[f->head];
    r->timestamp = ts;
    r->depth = depth;
    r->is_exit = is_exit;
    strncpy(r->func, func ? func : "", WUBU_FTRACE_FUNC_MAX - 1);
    r->func[WUBU_FTRACE_FUNC_MAX - 1] = '\0';
    f->head = (f->head + 1) % WUBU_FTRACE_CAP;
    if (f->count < WUBU_FTRACE_CAP) f->count++;
}

int wubu_ftrace_enter(wubu_ftrace_t *f, const char *func, uint64_t timestamp)
{
    if (!f) return -1;
    if (!f->enabled) return 0;
    record(f, func, timestamp, 0, f->cur_depth);
    f->cur_depth++;
    return 0;
}

int wubu_ftrace_exit(wubu_ftrace_t *f, const char *func, uint64_t timestamp)
{
    if (!f) return -1;
    if (!f->enabled) return 0;
    if (f->cur_depth > 0) f->cur_depth--;
    record(f, func, timestamp, 1, f->cur_depth);
    return 0;
}

int wubu_ftrace_read(const wubu_ftrace_t *f, int i, wubu_ftrace_rec_t *out)
{
    if (!f || !out) return -1;
    if (i < 0 || i >= f->count) return -1;
    /* oldest is at (head - count + CAP) % CAP */
    int start = (f->head - f->count + WUBU_FTRACE_CAP) % WUBU_FTRACE_CAP;
    int idx = (start + i) % WUBU_FTRACE_CAP;
    *out = f->ring[idx];
    return 0;
}

int wubu_ftrace_count(const wubu_ftrace_t *f)
{
    return f ? f->count : 0;
}

int wubu_ftrace_dump(const wubu_ftrace_t *f, char *buf, size_t cap)
{
    if (!f || !buf || cap == 0) return -1;
    size_t off = 0;
    int n = 0;
    for (int i = 0; i < f->count; i++) {
        wubu_ftrace_rec_t r;
        if (wubu_ftrace_read(f, i, &r) != 0) break;
        int w = snprintf(buf + off, cap - off,
                         "%llu %*s%s %s\n",
                         (unsigned long long)r.timestamp,
                         r.depth * 2, "",
                         r.is_exit ? "}" : "{",
                         r.func);
        if (w < 0 || (size_t)w >= cap - off) { buf[off ? off - 1 : 0] = '\0'; return (int)off; }
        off += (size_t)w;
        n++;
    }
    if (off < cap) buf[off] = '\0';
    return (int)off;
}
