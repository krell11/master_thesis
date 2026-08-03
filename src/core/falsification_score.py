"""Score one Policy response via Falsifier → RAG(E−) → Adjudicator → reward.

Used by offline full_pipeline and by verl custom reward (online GRPO).
"""

from __future__ import annotations

from typing import Callable

from src.core.adjudicator import build_adjudicator_prompt, parse_adjudicator_output
from src.core.falsifier import build_falsifier_prompt, parse_falsifier_output
from src.core.policy import parse_policy_output
from src.core.reward import PARSE_FAIL_PENALTY, compute_reward
from src.core.retriever import CorpusRetriever, format_retrieved_context

MAX_CONTEXT_CHARS = 6000
TOP_K_NEG = 3
ANTI_QUERIES_PER_CLAIM = 2

GenerateFn = Callable[[list[str]], list[str]]


def align_falsifier_claims(
    policy_claims: list[dict],
    falsifier_out: dict,
    *,
    anti_queries_per_claim: int = ANTI_QUERIES_PER_CLAIM,
) -> list[dict]:
    """Match anti_queries back onto policy claims by claim_id (fallback: order)."""
    by_id: dict[str, list[str]] = {}
    for c in falsifier_out.get("claims") or []:
        cid = str(c.get("claim_id", ""))
        aqs = [q.strip() for q in (c.get("anti_queries") or []) if str(q).strip()]
        by_id[cid] = aqs[:anti_queries_per_claim]

    aligned = []
    for i, pc in enumerate(policy_claims):
        cid = str(pc.get("claim_id", f"c{i}"))
        aqs = by_id.get(cid)
        if aqs is None:
            raw = falsifier_out.get("claims") or []
            aqs = []
            if i < len(raw):
                aqs = [
                    q.strip()
                    for q in (raw[i].get("anti_queries") or [])
                    if str(q).strip()
                ][:anti_queries_per_claim]
        if not aqs:
            aqs = [f"evidence against: {pc['text'][:120]}"]
        aligned.append(
            {
                "claim_id": cid,
                "text": pc["text"],
                "anti_queries": aqs,
            }
        )
    return aligned


def retrieve_counter_evidence(
    retriever: CorpusRetriever,
    anti_queries: list[str],
    *,
    top_k: int = TOP_K_NEG,
    exclude_paper_id: str | None = None,
) -> list[dict]:
    """Retrieve E− for anti-queries; optionally drop chunks from the source paper."""
    seen: set[str] = set()
    hits: list[dict] = []
    for q in anti_queries:
        for h in retriever.retrieve(q, top_k=top_k + 3):
            if exclude_paper_id and h["paper_id"] == exclude_paper_id:
                continue
            cid = h["chunk_id"]
            if cid in seen:
                continue
            seen.add(cid)
            item = dict(h)
            item["anti_query"] = q
            hits.append(item)
            if len(hits) >= top_k * max(len(anti_queries), 1):
                return hits
    return hits


def score_policy_response(
    *,
    question: str,
    paper_name: str,
    paper_id: str,
    solution_str: str,
    retriever: CorpusRetriever,
    tokenizer,
    falsifier_generate: GenerateFn,
    adjudicator_generate: GenerateFn,
    top_k_neg: int = TOP_K_NEG,
    exclude_source_paper: bool = True,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> dict:

    try:
        policy = parse_policy_output(solution_str)
    except Exception as exc:
        return {
            "score": PARSE_FAIL_PENALTY,
            "reward": PARSE_FAIL_PENALTY,
            "parse_ok": False,
            "parse_error": str(exc),
            "y": None,
            "claims": [],
            "reward_breakdown": {
                "n_claims": 0,
                "n_refuted": 0,
                "n_supported": 0,
                "n_insufficient": 0,
                "claim_rewards": [],
            },
        }

    y = policy["y"]
    claims = policy["claims"]

    falsifier_prompt = build_falsifier_prompt(
        paper_name,
        claims,
        tokenizer,
        question=question,
        answer=y,
    )
    try:
        fout = parse_falsifier_output(falsifier_generate([falsifier_prompt])[0])
    except Exception:
        fout = {"claims": []}
    aligned = align_falsifier_claims(claims, fout)

    exclude = paper_id if exclude_source_paper else None
    claims_ev = []
    adj_prompts = []
    for c in aligned:
        hits_neg = retrieve_counter_evidence(
            retriever,
            c["anti_queries"],
            top_k=top_k_neg,
            exclude_paper_id=exclude,
        )
        evidence_text = format_retrieved_context(
            hits_neg, max_chars=max_context_chars
        )
        claims_ev.append(
            {
                "claim_id": c["claim_id"],
                "text": c["text"],
                "anti_queries": c["anti_queries"],
                "evidence_neg": hits_neg,
                "evidence_text": evidence_text,
            }
        )
        adj_prompts.append(
            build_adjudicator_prompt(
                {"claim_id": c["claim_id"], "text": c["text"]},
                evidence_text,
                tokenizer,
                question=question,
                answer=y,
            )
        )

    if adj_prompts:
        adj_texts = adjudicator_generate(adj_prompts)
    else:
        adj_texts = []

    verdicts = []
    claims_audit = []
    for ce, raw in zip(claims_ev, adj_texts):
        v = parse_adjudicator_output(raw, claim_id=ce["claim_id"])
        verdicts.append(v)
        claims_audit.append(
            {
                "claim_id": ce["claim_id"],
                "text": ce["text"],
                "anti_queries": ce["anti_queries"],
                "evidence_neg": [
                    {
                        "paper_id": h["paper_id"],
                        "paper_name": h["paper_name"],
                        "section": h["section"],
                        "chunk_id": h["chunk_id"],
                        "score": h["score"],
                        "anti_query": h.get("anti_query"),
                        "text": h["text"][:400],
                    }
                    for h in ce["evidence_neg"]
                ],
                "verdict": v["verdict"],
                "rationale": v.get("rationale", ""),
            }
        )

    reward_info = compute_reward(verdicts)
    return {
        "score": float(reward_info["reward"]),
        "reward": float(reward_info["reward"]),
        "parse_ok": True,
        "parse_error": None,
        "y": y,
        "claims": claims_audit,
        "reward_breakdown": {
            "n_claims": reward_info["n_claims"],
            "n_refuted": reward_info["n_refuted"],
            "n_supported": reward_info["n_supported"],
            "n_insufficient": reward_info["n_insufficient"],
            "claim_rewards": reward_info["claim_rewards"],
        },
    }
