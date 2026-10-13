/* wubu_ns_iso.c -- OMARCHY #278 namespace isolation (mount/pid/net/user).
 * See wubu_ns_iso.h. Policy core, fail-closed. */
#include "wubu_ns_iso.h"

#define WUBU_NS_ALL (WUBU_NS_MOUNT|WUBU_NS_PID|WUBU_NS_NET|WUBU_NS_USER| \
                     WUBU_NS_UTS|WUBU_NS_IPC|WUBU_NS_CGROUP)

int wubu_ns_iso_init(wubu_ns_iso_t *n) {
    if (!n) return -1;
    n->private_ns = WUBU_NS_ALL;   /* fully isolated */
    n->shared_ns = 0;
    n->may_create_new = 0;         /* no new namespaces */
    return 0;
}

static int eq_ci(const char *a, const char *b) {
    while (*a && *b) {
        char x = *a, y = *b;
        if (x >= 'A' && x <= 'Z') x += 32;
        if (y >= 'A' && y <= 'Z') y += 32;
        if (x != y) return 0;
        a++; b++;
    }
    return *a == '\0' && *b == '\0';
}
static int truthy(const char *v) {
    return eq_ci(v, "yes") || eq_ci(v, "true") || eq_ci(v, "1");
}

/* Map a directive to the namespace kind it governs. */
static unsigned directive_ns(const char *directive) {
    if (eq_ci(directive, "PrivateMounts") || eq_ci(directive, "MountNamespace")) return WUBU_NS_MOUNT;
    if (eq_ci(directive, "PrivatePID") || eq_ci(directive, "PIDNamespace")) return WUBU_NS_PID;
    if (eq_ci(directive, "PrivateNetwork") || eq_ci(directive, "NetworkNamespace")) return WUBU_NS_NET;
    if (eq_ci(directive, "PrivateUsers") || eq_ci(directive, "UserNamespace")) return WUBU_NS_USER;
    if (eq_ci(directive, "PrivateIPC") || eq_ci(directive, "IPCNamespace")) return WUBU_NS_IPC;
    if (eq_ci(directive, "PrivateTmp")) return 0;  /* not a full namespace; ignore */
    return 0;
}

int wubu_ns_iso_parse(wubu_ns_iso_t *n, const char *directive, const char *value) {
    if (!n || !directive || !value) return -1;
    if (eq_ci(directive, "RestrictNamespaces")) {
        /* RestrictNamespaces=yes -> no new namespaces; =no -> may create. */
        n->may_create_new = truthy(value) ? 0 : 1;
        return 0;
    }
    unsigned kind = directive_ns(directive);
    if (kind == 0) return -1;   /* unknown directive */
    if (truthy(value)) {
        n->private_ns |= kind;
        n->shared_ns &= ~kind;
    } else {
        n->private_ns &= ~kind;
        n->shared_ns |= kind;
    }
    return 0;
}

int wubu_ns_is_private(const wubu_ns_iso_t *n, wubu_ns_kind_t kind) {
    if (!n) return 1;   /* fail-closed: unknown = isolated */
    return (n->private_ns & (unsigned)kind) ? 1 : 0;
}

int wubu_ns_may_create(const wubu_ns_iso_t *n, wubu_ns_kind_t kind) {
    if (!n) return 0;   /* fail-closed */
    (void)kind;
    return n->may_create_new;
}

double wubu_ns_isolation_score(const wubu_ns_iso_t *n) {
    if (!n) return 1.0;   /* fail-closed: fully isolated */
    int total = 7;
    int priv = 0;
    for (int i = 0; i < total; i++)
        if (n->private_ns & (1u << i)) priv++;
    return (double)priv / (double)total;
}
