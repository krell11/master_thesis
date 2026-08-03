
from __future__ import annotations

# Per-claim scores: refutation hurts more than support helps.
VERDICT_SCORE = {
    "refuted": -1.0,
    "supported": 0.5,
    "insufficient": 0.0,
}

PARSE_FAIL_PENALTY = -1.0


def claim_reward(verdict: str) -> float:
    return float(VERDICT_SCORE.get(verdict, 0.0))


def compute_reward(
    verdicts: list[dict],
    *,
    empty_claims_penalty: float = -0.5,
    all_insufficient_penalty: float = -0.1,
) -> dict:
    if not verdicts:
        return {
            "reward": empty_claims_penalty,
            "n_claims": 0,
            "n_refuted": 0,
            "n_supported": 0,
            "n_insufficient": 0,
            "claim_rewards": [],
        }

    claim_rewards = []
    counts = {"refuted": 0, "supported": 0, "insufficient": 0}
    for v in verdicts:
        verdict = v.get("verdict", "insufficient")
        if verdict not in counts:
            verdict = "insufficient"
        counts[verdict] += 1
        r = claim_reward(verdict)
        claim_rewards.append(
            {
                "claim_id": v.get("claim_id"),
                "verdict": verdict,
                "reward": r,
            }
        )

    reward = sum(cr["reward"] for cr in claim_rewards) / len(claim_rewards)
    if counts["refuted"] == 0 and counts["supported"] == 0:
        reward += all_insufficient_penalty

    return {
        "reward": float(reward),
        "n_claims": len(verdicts),
        "n_refuted": counts["refuted"],
        "n_supported": counts["supported"],
        "n_insufficient": counts["insufficient"],
        "claim_rewards": claim_rewards,
    }
