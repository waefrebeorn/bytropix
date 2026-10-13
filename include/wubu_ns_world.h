#ifndef WUBU_NS_WORLD_H
#define WUBU_NS_WORLD_H
#include "wubu_worldsim.h"
#include <stddef.h>

/* wubu_ns_world.h -- WORLDSIM #262 worldsim state at /n/world.
 *
 * Exposes the live worldsim state as 9P-readable files under /n/world:
 *   /n/world/steps     -- total world steps taken
 *   /n/world/bodies    -- number of bodies
 *   /n/world/ke        -- total kinetic energy
 *   /n/world/body/{i}/x, /y, /vx, /vy, /mass, /radius  -- per-body state
 * Each is rendered as a text line (9P file content) from the live world.
 * Deterministic, no allocation beyond the caller's buffer, fail-closed.
 * Real work (no stubs). */

/* Render a /n/world stat file's content into out (NUL-terminated). `relpath`
 * is the path after "/n/world/" (e.g. "steps", "body/0/x"). Returns the
 * number of bytes written (excluding NUL), or -1 if the path is unknown. */
int wubu_ns_world_read(const wubu_world_t *w, const char *relpath,
                       char *out, size_t outsz);

#endif
