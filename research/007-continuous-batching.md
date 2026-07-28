# 007 — Continuous (iteration-level) batching

Source: Anyscale "Continuous Batching" (23× throughput); vLLM continuous batching
(iteration-level scheduling); Spheron TL;DR table (+2–3× vs static batching).
Also chunked prefill, disaggregated prefill/decode (D03).

## Core idea
Instead of waiting for a whole batch of requests to finish before starting new
ones, schedule at the *iteration* (token) level: when any request finishes its
token, a waiting request slots into the freed GPU/CPU position immediately. This
takes GPU/CPU utilization from 30–60% to 80–95% under variable-length traffic.
For a CPU engine the win is the same: keep the GEMV busy across requests instead
of stalling on the longest sequence.

## Triple-DA
- P1 correctness: batching is a scheduling concern; each request's KV + weights
  stay separate. Token order per request is preserved. ✓
- P2 privacy: local scheduler. ✓
- P3 robustness: a runaway long request must not starve others — cap tokens/request
  per scheduling round (fairness), and bound KV memory per request.

## Implementation plan
- `wubu_scheduler.c/.h`: a request queue; each engine step runs 1 token for up to
  N in-flight requests (N = budget). New requests join mid-step.
- The existing single-sequence forward becomes "forward one token for request r".
- Combine with D04 (chunked prefill) so a long prompt doesn't block decodes.

## Test oracle
- Submit 8 requests of varying length; assert all complete and throughput
  (tok/s aggregate) > single-request baseline × ~2.
- Assert no request starves (max latency bounded).
