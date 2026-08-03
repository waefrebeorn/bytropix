/* test_traj_sft.c -- the masked-observation SFT converter: the masks must
 * follow the Hermes/Orchard doctrine (assistant/think/tool_call train;
 * user/obs/tool_resp are context), the parse must be deterministic, and
 * the train-fraction must be measurable. */
#include <stdio.h>
#include <string.h>
#include "wubu_traj_sft.h"

int main(void)
{
    /* a typical 3-cycle agent trajectory (one segment per line) */
    char traj[] =
        "[user] book a flight to Denver under 500 dollars\n"
        "[assistant] I will search for flights under $500.\n"
        "[tool_call] search_flights(dest=DEN, max=500)\n"
        "[tool_resp] 3 flights found: UA 1204 $312, UA 998 $455, WN 88 $498\n"
        "[think] UA 1204 fits the budget; the user wants a direct flight.\n"
        "[assistant] I found a direct UA flight at $312 -- shall I book it?\n"
        "[user] yes, book it\n"
        "[tool_call] book_flight(UA1204)\n"
        "[tool_resp] confirmed, booking ref AB123\n"
        "[assistant] Done -- your flight is booked (ref AB123).\n"
        "[obs] the booking database now shows AB123 under your account\n";

    wubu_sft_seg_t segs[16];
    int n = wubu_traj_sft_convert(traj, segs, 16);
    if (n != 11) { printf("  parse count %d (expected 11) FAIL\n", n); return 1; }

    /* the expected mask pattern: user=0, assistant=1, tool_call=1,
     * tool_resp=0, think=1, ... */
    int expect_train[] = {0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0};
    int expect_type[] = {WUBU_SEG_USER, WUBU_SEG_ASSISTANT, WUBU_SEG_TOOL_CALL,
                         WUBU_SEG_TOOL_RESP, WUBU_SEG_THINK, WUBU_SEG_ASSISTANT,
                         WUBU_SEG_USER, WUBU_SEG_TOOL_CALL, WUBU_SEG_TOOL_RESP,
                         WUBU_SEG_ASSISTANT, WUBU_SEG_OBS};
    int ok = 1;
    for (int i = 0; i < n; i++) {
        if (segs[i].train != expect_train[i] ||
            segs[i].type != expect_type[i]) {
            printf("  seg %d: type %d train %d (expected %d/%d) FAIL\n",
                   i, segs[i].type, segs[i].train,
                   expect_type[i], expect_train[i]);
            ok = 0;
        }
    }
    /* the body text must be stripped of the tag */
    if (strcmp(segs[1].text, "I will search for flights under $500.") != 0) {
        printf("  body strip FAIL: '%s'\n", segs[1].text);
        ok = 0;
    }
    /* the train fraction: 6 of 11 segments train (assistant/think/tool_call) */
    float f = wubu_traj_sft_train_frac(segs, n);
    if (f < 0.53f || f > 0.56f) {
        printf("  train frac %.3f (expected ~0.545) FAIL\n", f);
        ok = 0;
    }
    /* determinism: converting the same buffer twice gives the same masks */
    wubu_sft_seg_t segs2[16];
    int n2 = wubu_traj_sft_convert(traj, segs2, 16);
    if (n2 != n) { printf("  determinism FAIL\n"); ok = 0; }
    for (int i = 0; i < n; i++)
        if (segs2[i].train != segs[i].train ||
            strcmp(segs2[i].text, segs[i].text) != 0) {
            printf("  determinism seg %d FAIL\n", i);
            ok = 0;
        }
    /* the empty/bad input */
    if (wubu_traj_sft_convert("no tags here\n", segs, 16) != 0) {
        printf("  untagged input should yield 0 segments FAIL\n");
        ok = 0;
    }
    wubu_traj_sft_segs_free(segs, n);
    wubu_traj_sft_segs_free(segs2, n2);
    printf("  segments %d, train frac %.3f  %s\n", n, f, ok ? "PASS" : "FAIL");
    printf("%s\n", ok ? "ALL TRAJ-SFT TESTS PASSED" : "TRAJ-SFT FAILURES");
    return ok ? 0 : 1;
}
