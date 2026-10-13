/* wubu_volmixer.c -- Win98/XP per-application volume mixer.
 * See wubu_volmixer.h. Real, deterministic per-app volume controller. */

#include "wubu_volmixer.h"
#include <string.h>

static int clamp100(int v) { return v < 0 ? 0 : (v > 100 ? 100 : v); }

int wubu_vm_init(wubu_volmixer_t *vm) {
    if (!vm) return -1;
    memset(vm, 0, sizeof(*vm));
    vm->master = 80;
    vm->master_muted = 0;
    return 0;
}

int wubu_vm_set_master(wubu_volmixer_t *vm, int volume) {
    if (!vm) return -1;
    vm->master = clamp100(volume);
    return 0;
}

int wubu_vm_set_master_muted(wubu_volmixer_t *vm, int muted) {
    if (!vm) return -1;
    vm->master_muted = muted ? 1 : 0;
    return 0;
}

static int find(const wubu_volmixer_t *vm, const char *name) {
    if (!vm || !name) return -1;
    for (int i = 0; i < vm->n_channels; i++)
        if (strcmp(vm->channels[i].name, name) == 0) return i;
    return -1;
}

int wubu_vm_add_channel(wubu_volmixer_t *vm, const char *name, int volume) {
    if (!vm || !name) return -1;
    if (find(vm, name) >= 0) return -1;   /* dup */
    if (vm->n_channels >= WUBU_VM_MAX_CHANNELS) return -1;
    wubu_vm_channel_t *c = &vm->channels[vm->n_channels];
    memset(c, 0, sizeof(*c));
    strncpy(c->name, name, WUBU_VM_NAME_MAX - 1);
    c->volume = clamp100(volume);
    c->muted = 0;
    return vm->n_channels++;
}

int wubu_vm_set_volume(wubu_volmixer_t *vm, const char *name, int volume) {
    if (!vm) return -1;
    int i = find(vm, name);
    if (i < 0) return -1;
    vm->channels[i].volume = clamp100(volume);
    return 0;
}

int wubu_vm_set_muted(wubu_volmixer_t *vm, const char *name, int muted) {
    if (!vm) return -1;
    int i = find(vm, name);
    if (i < 0) return -1;
    vm->channels[i].muted = muted ? 1 : 0;
    return 0;
}

int wubu_vm_effective(const wubu_volmixer_t *vm, const char *name) {
    int i = find(vm, name);
    if (i < 0) return -1;
    if (vm->master_muted || vm->channels[i].muted) return 0;
    return vm->master * vm->channels[i].volume / 100;
}

int wubu_vm_overall(const wubu_volmixer_t *vm) {
    if (!vm) return -1;
    if (vm->master_muted || vm->n_channels == 0) return 0;
    int max = 0;
    for (int i = 0; i < vm->n_channels; i++) {
        if (vm->channels[i].muted) continue;
        int e = vm->master * vm->channels[i].volume / 100;
        if (e > max) max = e;
    }
    return max;
}
