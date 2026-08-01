/*
 * wubu_capzero.h -- Capability/Zero-Trust kernel for the AGI-OS (AF02-AF04).
 */
#ifndef WUBU_CAPZERO_H
#define WUBU_CAPZERO_H

#include <stdint.h>
#include <stdlib.h>

#define WUBU_CAP_MAX_TOOLS 64
#define WUBU_CAP_NAME_LEN  32

/* AF02 deny-by-default tool registry. */
typedef struct wubu_capset wubu_capset_t;
wubu_capset_t *wubu_capset_create(void);
void  wubu_capset_destroy(wubu_capset_t *c);
int   wubu_cap_grant(wubu_capset_t *c, const char *tool);
int   wubu_cap_check(const wubu_capset_t *c, const char *tool); /* 1=allowed, 0=denied */

/* AF04 non-human identity token. */
uint64_t wubu_nhi_issue(const char *agent_id, const char *secret);
int      wubu_nhi_valid(uint64_t tok);

/* AF03 encrypted agent memory at rest (CTR stream; symmetric in-place). */
void wubu_mem_crypt(uint64_t key, uint64_t nonce, unsigned char *buf, size_t len);

#endif
