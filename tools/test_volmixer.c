/* test_volmixer.c -- Win98/XP per-application volume mixer tests. */
#include "wubu_volmixer.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_volmixer_t vm;

    CHECK(wubu_vm_init(&vm) == 0);
    CHECK(vm.master == 80);
    CHECK(vm.master_muted == 0);
    CHECK(wubu_vm_init(NULL) == -1);

    /* master clamp */
    CHECK(wubu_vm_set_master(&vm, 120) == 0);
    CHECK(vm.master == 100);
    CHECK(wubu_vm_set_master(&vm, -5) == 0);
    CHECK(vm.master == 0);
    CHECK(wubu_vm_set_master(&vm, 50) == 0);
    CHECK(vm.master == 50);
    CHECK(wubu_vm_set_master(NULL, 50) == -1);
    CHECK(wubu_vm_set_master_muted(NULL, 1) == -1);

    /* add channels */
    CHECK(wubu_vm_add_channel(&vm, "system", 100) == 0);
    CHECK(wubu_vm_add_channel(&vm, "firefox", 60) == 1);
    CHECK(wubu_vm_add_channel(&vm, "game", 30) == 2);
    CHECK(vm.n_channels == 3);
    CHECK(wubu_vm_add_channel(&vm, "system", 50) == -1);   /* dup */
    CHECK(wubu_vm_add_channel(&vm, NULL, 50) == -1);
    CHECK(wubu_vm_add_channel(NULL, "x", 50) == -1);

    /* effective = master * channel / 100 (master 50) */
    CHECK(wubu_vm_effective(&vm, "system") == 50);    /* 50*100/100 */
    CHECK(wubu_vm_effective(&vm, "firefox") == 30);   /* 50*60/100 */
    CHECK(wubu_vm_effective(&vm, "game") == 15);      /* 50*30/100 */
    CHECK(wubu_vm_effective(&vm, "nope") == -1);

    /* overall = max effective = 50 */
    CHECK(wubu_vm_overall(&vm) == 50);

    /* mute a channel */
    CHECK(wubu_vm_set_muted(&vm, "system", 1) == 0);
    CHECK(wubu_vm_effective(&vm, "system") == 0);     /* muted -> 0 */
    CHECK(wubu_vm_overall(&vm) == 30);                /* now firefox (60) is max */
    CHECK(wubu_vm_set_muted(&vm, "system", 0) == 0);
    CHECK(wubu_vm_effective(&vm, "system") == 50);
    CHECK(wubu_vm_set_muted(&vm, "nope", 1) == -1);
    CHECK(wubu_vm_set_muted(NULL, "system", 1) == -1);

    /* master mute zeroes everything */
    CHECK(wubu_vm_set_master_muted(&vm, 1) == 0);
    CHECK(wubu_vm_effective(&vm, "system") == 0);
    CHECK(wubu_vm_effective(&vm, "firefox") == 0);
    CHECK(wubu_vm_overall(&vm) == 0);
    CHECK(wubu_vm_set_master_muted(&vm, 0) == 0);
    CHECK(wubu_vm_effective(&vm, "system") == 50);

    /* set_volume clamp + effect */
    CHECK(wubu_vm_set_volume(&vm, "game", 200) == 0);
    CHECK(vm.channels[2].volume == 100);   /* clamped */
    CHECK(wubu_vm_effective(&vm, "game") == 50);   /* 50*100/100 */
    CHECK(wubu_vm_set_volume(&vm, "game", -1) == 0);
    CHECK(vm.channels[2].volume == 0);
    CHECK(wubu_vm_effective(&vm, "game") == 0);
    CHECK(wubu_vm_set_volume(&vm, "nope", 50) == -1);
    CHECK(wubu_vm_set_volume(NULL, "system", 50) == -1);

    /* overall with all channels muted = 0 */
    CHECK(wubu_vm_set_muted(&vm, "system", 1) == 0);
    CHECK(wubu_vm_set_muted(&vm, "firefox", 1) == 0);
    CHECK(wubu_vm_set_muted(&vm, "game", 1) == 0);
    CHECK(wubu_vm_overall(&vm) == 0);

    /* empty mixer: overall 0 */
    wubu_volmixer_t empty;
    CHECK(wubu_vm_init(&empty) == 0);
    CHECK(wubu_vm_overall(&empty) == 0);
    CHECK(wubu_vm_effective(&empty, "x") == -1);

    /* bad args */
    CHECK(wubu_vm_effective(NULL, "x") == -1);
    CHECK(wubu_vm_effective(&vm, NULL) == -1);
    CHECK(wubu_vm_overall(NULL) == -1);

    if (fails == 0) printf("ALL VOLMIXER TESTS PASSED\n");
    else printf("%d VOLMIXER TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
