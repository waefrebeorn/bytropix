#ifndef WUBU_NS_ISO_H
#define WUBU_NS_ISO_H
#include <stddef.h>

/* wubu_ns_iso.h -- OMARCHY #278 namespace isolation (mount/pid/net/user).
 *
 * Models the Linux namespace set that isolates a service (the unshare(2)
 * flags systemd/RestrictNamespaces uses) and computes an isolation verdict:
 * which namespaces a unit is confined to, whether it may create NEW
 * namespaces (mount/pid/net/user) to escape, and a human-readable summary.
 * Pure-C policy core: the actual unshare() is the kernel's job; this module
 * owns the decision of which namespaces are private vs shared and whether
 * new-namespace creation is permitted. Fail-closed: an unknown/empty profile
 * is treated as most-isolated. Deterministic, no allocation, real work. */

/* Namespace kinds (bitmask). Mirrors the CLONE_NEW* flags conceptually. */
typedef enum {
    WUBU_NS_MOUNT = 1u<<0,   /* mount namespace (filesystem view) */
    WUBU_NS_PID   = 1u<<1,   /* pid namespace (process view) */
    WUBU_NS_NET   = 1u<<2,   /* net namespace (network stack) */
    WUBU_NS_USER  = 1u<<3,   /* user namespace (uid mapping) */
    WUBU_NS_UTS   = 1u<<4,   /* uts namespace (hostname) */
    WUBU_NS_IPC   = 1u<<5,   /* ipc namespace (sysv/posix ipc) */
    WUBU_NS_CGROUP= 1u<<6,   /* cgroup namespace */
} wubu_ns_kind_t;

typedef struct {
    unsigned private_ns;    /* namespaces the unit gets a PRIVATE copy of */
    unsigned shared_ns;     /* namespaces the unit SHARES with the host */
    int      may_create_new;/* 1 = may unshare additional namespaces */
} wubu_ns_iso_t;

/* Initialise a fully-isolated profile (all namespaces private, no new). */
int wubu_ns_iso_init(wubu_ns_iso_t *n);
/* Parse a "Directive=value" (RestrictNamespaces, PrivateMounts, PrivateNetwork,
 * PrivateUsers, PrivatePID, PrivateIPC). Returns 0/-1. */
int wubu_ns_iso_parse(wubu_ns_iso_t *n, const char *directive, const char *value);
/* Is a given namespace private to the unit? 1/0. */
int wubu_ns_is_private(const wubu_ns_iso_t *n, wubu_ns_kind_t kind);
/* May the unit create a NEW namespace of this kind (escape vector)? Fail-closed. */
int wubu_ns_may_create(const wubu_ns_iso_t *n, wubu_ns_kind_t kind);
/* Isolation score: fraction of the 7 namespaces that are private, in [0,1]. */
double wubu_ns_isolation_score(const wubu_ns_iso_t *n);

#endif
