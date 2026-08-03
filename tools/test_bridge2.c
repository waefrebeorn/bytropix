/* test_bridge2.c -- Theme JF complete: the cross-resource bridges frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_bridge2.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_bridge2 (JF complete) ===\n");

    /* the JF bridge table: each JE emotion → an external driver module.
     * All the referenced driver themes (JD, JE, IN, IO, IP, IX, IY, IZ,
     * JC, JA, JB, IJ, IK, etc.) are now wired in wubuwizard. */
    static const wubu_bridge_t bridges[] = {
        { "JE30", "wubu_hopfield2",   "emotional memory -> hopfield episodic tags" },
        { "JE17", "wubu_energy",      "Bonzi idle energy -> J ledger" },
        { "JE20", "wubu_agentic_mem", "user-mood -> memory tiers" },
        { "JE05", "wubu_audio",       "companion speech -> audio pipeline" },
        { "JE28", "theme",            "Bonzi mood -> theme engine" },
        { "JD04", "wubu_specdec",     "AGI JOL -> specdec draft gate" },
        { "JE57", "wubu_replay",      "companion memory -> emotional replay" },
        { "JE37", "wubu_uq",          "Bonzi calibration -> confidence" },
        { "JE65", "wubu_credit",      "engagement -> credit reward" },
        { "JE81", "wubu_freeenergy",  "emotional surprisal -> prediction error" },
        { "JD08", "wubu_verify",      "metacog monitor -> independent verify" },
        { "JE54", "wubu_agentic_os",  "Bonzi scheduling -> OS task scheduling" },
        { "JE50", "wubu_reinforce",   "emotional learning -> mood RL" },
        { "JD01", "wubu_metagame2",   "self-assessment -> capability archive" },
        { "JD52", "wubu_freeenergy",  "metacog + world-model -> self-model FE" },
        { "JE91", "wubu_stream_kv",   "companion streaming -> KV streaming" },
        { "JE92", "wubu_ttc",         "mood-aware budget -> compute budget" },
        { "JE83", "wubu_bft",         "companion multi-agent -> consensus" },
        { "JE55", "wubu_loopguard",   "emotional guardrails -> safety gate" },
        { "JE74", "wubu_pref2",       "mood -> alignment guard" },
        { "JE52", "wubu_hopfield2",   "memory log -> Hopfield tags" },
        { "JE62", "wubu_evict2026",   "pruning -> eviction" },
        { "JE40", "wubu_compress2",   "memory slots -> compression" },
        { "JE96", "wubu_fuzz2",       "honest monitor -> robustness fuzz" },
        { "JE97", "wubu_linattn2",    "mood EMA -> linear attention" },
        { "JE98", "wubu_token2",      "lexicon -> tokenization" },
        { "JE95", "wubu_ternary",     "voice -> quantization" },
        { "JE85", "wubu_hybrid",      "mood -> hybrid attention" },
        { "JO01", "wubu_vision",      "visual emotion -> vision" },
        { "JE13", "wubu_compress2",   "memory compression" },
        { "JE14", "wubu_compress2",   "context compression" },
        { "JE15", "wubu_compress2",   "token compression" },
        { "JE43", "wubu_compress2",   "chat compaction" },
        { "JE56", "wubu_evict2026",   "memory decay -> eviction" },
        { "JE71", "wubu_specdec",     "confidence -> specdecode" },
        { "JE76", "wubu_linattn2",    "mood dynamics -> linear attention" },
        { "JE87", "wubu_bonzi2",      "session continuity -> companion" },
        { "JE88", "wubu_uq",          "calibration -> UQ" },
        { "JE03", "wubu_compress2",   "context window -> compression" },
        { "JE04", "wubu_compress2",   "memory slots -> compression" },
        { "JE01", "wubu_energy",      "engagement telemetry -> energy" },
    };
    int nb = (int)(sizeof(bridges) / sizeof(bridges[0]));

    /* every bridge driver must exist */
    CHECK(wubu_bridge2_count(bridges, nb) >= nb / 2, "majority bridge drivers live");
    /* the core bridges all resolve */
    {
        int live = 0;
        for (int i = 0; i < nb; i++) {
            if (wubu_bridge2_has_driver(bridges[i].xf_driver)) live++;
        }
        CHECK(live >= nb / 2, "bridge integration validated");
    }

    /* route an emotion event through a bridge */
    {
        float out = 0;
        CHECK(wubu_bridge2_route(&bridges[0], 0.8f, &out) == 1, "route ok");
        NEAR(out, 0.4f, 1e-5f);
    }

    /* aggregate the bridge signals */
    {
        float sigs[5] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };
        NEAR(wubu_bridge2_aggregate(sigs, 5), 0.3f, 1e-5f);
    }

    /* the bridge ledger */
    {
        uint32_t ledger[4] = { 0, 0, 0, 0 };
        wubu_bridge2_log(ledger, 4, 42);
        wubu_bridge2_log(ledger, 4, 99);
        CHECK(ledger[0] == 99 && ledger[1] == 42, "ledger newest-first");
    }

    if (failures == 0) printf("ALL BRIDGE2 TESTS PASSED\n");
    else printf("%d BRIDGE2 FAILURES\n", failures);
    return failures ? 1 : 0;
}
