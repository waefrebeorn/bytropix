#ifndef WUBU_VOLMIXER_H
#define WUBU_VOLMIXER_H

/* wubu_volmixer.h -- Win98/XP per-application volume mixer.
 *
 * The system-tray volume control generalized to per-application channels: a
 * master volume plus a set of app channels each with its own volume (0-100)
 * and mute flag. The effective output of a channel is master * channel (both
 * scaled), and 0 if either the master or the channel is muted. Deterministic,
 * no stubs. */

#include <stddef.h>

#define WUBU_VM_MAX_CHANNELS 32
#define WUBU_VM_NAME_MAX 32

typedef struct {
    char name[WUBU_VM_NAME_MAX];
    int volume;      /* 0..100 */
    int muted;
} wubu_vm_channel_t;

typedef struct {
    int master;      /* 0..100 */
    int master_muted;
    wubu_vm_channel_t channels[WUBU_VM_MAX_CHANNELS];
    int n_channels;
} wubu_volmixer_t;

/* Initialise (master 80, unmuted, no channels). */
int wubu_vm_init(wubu_volmixer_t *vm);

/* Set the master volume (clamped 0..100) / mute. */
int wubu_vm_set_master(wubu_volmixer_t *vm, int volume);
int wubu_vm_set_master_muted(wubu_volmixer_t *vm, int muted);

/* Add an app channel. Returns index or -1 (dup/full). */
int wubu_vm_add_channel(wubu_volmixer_t *vm, const char *name, int volume);

/* Set a channel's volume (clamped 0..100) / mute by name. Returns 0/-1. */
int wubu_vm_set_volume(wubu_volmixer_t *vm, const char *name, int volume);
int wubu_vm_set_muted(wubu_volmixer_t *vm, const char *name, int muted);

/* Effective output (0..100) of a channel = master * channel / 100, 0 if
 * either is muted. Returns 0..100, -1 on unknown channel. */
int wubu_vm_effective(const wubu_volmixer_t *vm, const char *name);

/* Overall output (0..100): the max effective across channels, 0 if master
 * muted or no channels. */
int wubu_vm_overall(const wubu_volmixer_t *vm);

#endif
