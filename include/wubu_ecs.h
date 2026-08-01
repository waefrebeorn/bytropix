/*
 * wubu_ecs.h — ECS-style component store for engine state (doc C06).
 *
 * The inference engine has many interchangeable stateful subsystems (KV cache,
 * SSM states, MoE router, speculative decoder, the HW-accel wiring, etc.). An
 * "ECS" (Entity-Component-System) store gives each subsystem a named, typed
 * component slot it can register/snapshot/restore. This is the engine-state
 * analog of an entity-component store: one "engine entity" owns many components.
 * Ties to 001 (adaptive KV) and 006 (cache-line) — they register their state
 * here so a single snapshot/restore covers the whole engine. Pure C11.
 */
#ifndef WUBU_ECS_H
#define WUBU_ECS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_ecs wubu_ecs_t;

/* Create a store with capacity for `cap` components. */
wubu_ecs_t *wubu_ecs_create(int cap);
void wubu_ecs_free(wubu_ecs_t *e);

/* Register a named component of `size` bytes. Returns a stable component id
 * (>=0), or -1 if full / name clash. `dtor` (optional) is called on free. */
int wubu_ecs_add(wubu_ecs_t *e, const char *name, size_t size, void (*dtor)(void *));
/* Find a component id by name (or -1). */
int wubu_ecs_find(wubu_ecs_t *e, const char *name);

/* Get a typed pointer to a component's storage (NULL if absent). */
void *wubu_ecs_get(wubu_ecs_t *e, int cid);
/* Get component size (0 if absent). */
size_t wubu_ecs_size(wubu_ecs_t *e, int cid);
/* Get name (NULL if absent). */
const char *wubu_ecs_name(wubu_ecs_t *e, int cid);

/* Snapshot all component bytes into a contiguous buffer (caller frees with
 * free()). *out_bytes set to total. Returns NULL on alloc failure. */
uint8_t *wubu_ecs_snapshot(wubu_ecs_t *e, size_t *out_bytes);
/* Restore from a snapshot buffer (must match current component set by name+
 * size). Returns 0 on success, -1 on mismatch. */
int wubu_ecs_restore(wubu_ecs_t *e, const uint8_t *buf, size_t bytes);

/* Number of registered components. */
int wubu_ecs_count(wubu_ecs_t *e);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_ECS_H */
