# Multi-Agent Consensus + Inter-Agent Auth — 7-hop KB sweep
## DD axis: agents that agree (BFT) + prove identity (zero-trust), at home

> Each stone seeds the next hop. Target: map the distributed-trust substrate
> WuBuOS lacks — agents that can disagree safely and verify each other.

## Hop 1: Byzantine Fault Tolerance (BFT) consensus
Classical BFT: n = 3f+1 nodes tolerate f Byzantine (malicious) nodes.
Core result: 3n+1 is necessary and sufficient (Lamport, Pease, Shostak).
Threshold: 1/3 of nodes can be malicious. Multi-threshold BFT (MT-BFT) separates
fault tolerance for safety (≥2/3) vs liveness (≥1/3).
At home: our CoAgent coordination lock/unlock is a single-agent mechanism.
We need multi-agent: when `n` agent instances propose different configurations,
they must converge despite up to f malicious/lying proposers.

## Hop 2: Multi-round voting + threshold cryptography
3RVAV (Three-Round Voting, 2025): proof-of-stake + 3-round randomized voting.
Two-fold BFT: tolerate Byzantine + a-b-c faults beyond 1/3 threshold.
Multi-round voting: propose → prevote → precommit → commit (Tendermint-style).
Threshold cryptography: BLS signatures, each node signs → aggregate.
At home: we need a lightweight BFT that doesn't need a blockchain. We can
implement a simplified 3-round voting protocol (propose/accept/commit) with
a 2/3+1 threshold for small agent swarms (n ≤ 16). No crypto — just voting
with identity verification (the next hop).

## Hop 3: Inter-agent authentication (zero-trust, SPIFFE-based)
Zero Trust for agents: verify explicitly every hop, never implicit trust.
SPIFFE/SVID: workload identity via certs, not static API keys.
Behavioral identity: third layer — verify the agent's behavior matches its
declared purpose (beyond crypto identity).
At home: each CoAgent gets a SPIFFE-style identity (name + capability set).
Before an agent's vote counts, its identity is verified via a capability
attestation (signed by the orchestrator). Then its vote is tallied in BFT.

## Hop 4: Semantic consensus (AgentChain-style)
AgentChain: distributed semantic consensus. Instead of "the same input", agents
reason over different subsets and converge on semantic agreement.
Smart contracts as semantic signalling: agents post claims to a shared ledger,
other agents verify, disputes escalate. Tendermint consensus gadget ensures
agreement even with up to 1/3 malicious agents.
At home: each agent proposes a "claim" (e.g. "config X achieves 27 tok/s").
Other agents verify the claim (re-run the sweep, check tok/s). Disagreements
→ evidence submission → majority wins with 2/3+1 threshold.

## Hop 5: Distributed capability-based trust (DID + verifiable credentials)
Decentralized identity (DID): each agent has a self-certifying identity.
Verifiable credentials: attestations about the agent (e.g. "can read KV cache,
can spawn subprocesses"). No shared secrets — each agent issues and verifies
credentials cryptographically.
At home: we implement a simplified DID-like system — each agent has a
unique ID + capability set, signed by the orchestrator (WuBuOS kernel).
The signed capability set is the "verifiable credential." Agents present
their credential before participating in consensus.

## Hop 6: Fraud detection + dispute resolution
Byzantine agents may lie about their claims. Fraud detection: cross-check
results across agents, flag statistical outliers. Dispute resolution:
when agents disagree, they submit evidence (logs, measurements), and the
consensus layer adjudicates via majority vote or proof-of-work.
At home: when one CoAgent claims 27 tok/s but others see 15, the outlier
is flagged. The dispute is resolved by re-running with a median config,
and the fraudulent agent's vote weight is reduced (trust decay).

## Hop 7: Integration with WuBuOS substrate
The multi-agent consensus stack:
  1. Each CoAgent has a wubu_identity_t (ID + capabilities + signature)       [DD03]
  2. Agents propose claims (config + metrics)                                 [DD04]
  3. BFT voting: n agents, 2/3+1 threshold, 3 rounds (propose/accept/commit)  [DD01]
  4. Threshold signing: aggregate agent signatures for commit                 [DD02]
  5. Fraud detection: outlier detection + dispute resolution                  [DD05]
  6. Trust decay: repeat offenders lose voting weight                         [DD06]
  7. Consensus result feeds the DGM archive + recursive_optimize              [DD07]
  8. Inter-agent auth gates all inter-agent RPC (zero-trust)                  [DD03]

Closed-loop: agents collaborate truthfully, fraud is detected and penalized,
and the AGI-OS operator gets consensus-guaranteed configurations.

## Gap mapping
- DD01 BFT consensus (3-round voting, 2/3+1 threshold) `wired` (wubu_bft.c)
- DD02 Threshold signing (aggregate agent signatures) `wired` (wubu_threshsig.c)
- DD03 Inter-agent identity + zero-trust auth `wired` (wubu_agentid.c)
- DD04 Semantic consensus (claim + verify + dispute) `wired` (wubu_semcons.c)
- DD05 Fraud detection (outlier + dispute resolution) `wired` (wubu_fraud.c)
- DD06 Trust decay (repeat offender penalty) `wired` (test_multiconsensus)
- DD07 Integration: consensus → DGM archive + recursive_optimize `wired` (test_multiconsensus)
