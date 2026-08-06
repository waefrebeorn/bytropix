/* wubu_adapter_compat.h — compatibility shims for adapter versions
 *
 * Bridges between adapter API versions so old components can work with
 * new ones. The "shim" layer: when the model core upgrades from
 * attn_v1 to attn_v2, v1 adapters still work through a v1→v2 shim.
 *
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_ADAPTER_COMPAT_H
#define WUBU_ADAPTER_COMPAT_H

#include "wubu_adapter.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Register a compatibility shim between old and new adapter names.
 * When lookup for old_name fails, the shim redirects to new_name. */
int wubu_adapter_register_shim(const char *old_name, const char *new_name);

/* Like wubu_adapter_lookup but checks compat shims on miss. */
wubu_adapter_t *wubu_adapter_lookup_compat(const char *name);

/* Check if a shim exists for the given name. */
int wubu_adapter_has_shim(const char *name);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_ADAPTER_COMPAT_H */
