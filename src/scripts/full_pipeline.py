"""Full falsification pipeline (offline):
  x → RAG(E+) → Policy(y, claims)
    → Falsifier(anti-queries)
    → RAG(E−)
    → Adjudicator(verdicts)
    → asymmetric reward r
    → audit JSONL
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.core.adjudicator import build_adjudicator_prompt, parse_adjudicator_output
from src.core.falsifier import build_falsifier_prompt, parse_falsifier_output
from src.core.policy import build_policy_prompt, parse_policy_output
from src.core.reward import compute_reward
from src.core.retriever import CorpusRetriever, format_retrieved_context
from src.utils import load_jsonl, save_jsonl

model_path = str(ROOT / "models")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

MAX_CONTEXT_CHARS = 6000
TOP_K_POS = 5
TOP_K_NEG = 3
ANTI_QUERIES_PER_CLAIM = 2


def _align_falsifier_claims(policy_claims: list[dict], falsifier_out: dict) -> list[dict]:
    """Match anti_queries back onto policy claims by claim_id (fallback: order)."""
    by_id = {}
    for c in falsifier_out.get("claims") or []:
        cid = str(c.get("claim_id", ""))
        aqs = [q.strip() for q in (c.get("anti_queries") or []) if str(q).strip()]
        by_id[cid] = aqs[:ANTI_QUERIES_PER_CLAIM]

    aligned = []
    for i, pc in enumerate(policy_claims):
        cid = str(pc.get("claim_id", f"c{i}"))
        aqs = by_id.get(cid)
        if aqs is None:
            # fallback by index
            raw = (falsifier_out.get("claims") or [])
            aqs = []
            if i < len(raw):
                aqs = [
                    q.strip()
                    for q in (raw[i].get("anti_queries") or [])
                    if str(q).strip()
                ][:ANTI_QUERIES_PER_CLAIM]
        if not aqs:
            # last resort: searchable negation from claim text
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


def main(
    generated_path: str,
    rag_path: str,
    outputs_path: str,
    max_examples: int | None = None,
    top_k_pos: int = TOP_K_POS,
    top_k_neg: int = TOP_K_NEG,
    exclude_source_paper: bool = True,
):
    rag_dataset = load_jsonl(rag_path)
    generated = load_jsonl(generated_path)

    print(f"Building BM25 index over {len(rag_dataset)} papers...")
    retriever = CorpusRetriever(rag_dataset)
    print(f"Indexed {len(retriever.chunks)} chunks")

    policy_params = SamplingParams(
        n=1, top_p=0.9, temperature=0.2, seed=999, max_tokens=512
    )
    falsifier_params = SamplingParams(
        n=1, top_p=0.9, temperature=0.3, seed=999, max_tokens=512
    )
    adjudicator_params = SamplingParams(
        n=1, top_p=1.0, temperature=0.0, seed=999, max_tokens=128
    )

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.75,
        max_model_len=4096,
        max_num_seqs=8,
        language_model_only=True,
        trust_remote_code=True,
    )

    examples = []
    for row in generated:
        paper_id = row["id"]
        paper_name = row.get("title") or row.get("paper_name") or ""
        question = (row.get("qas") or row.get("question") or "").strip()
        question_original = (row.get("question") or question).strip()
        if not question:
            continue

        retrieved_pos = retriever.retrieve(question, top_k=top_k_pos)
        context = format_retrieved_context(retrieved_pos, max_chars=MAX_CONTEXT_CHARS)
        examples.append(
            {
                "id": paper_id,
                "paper_name": paper_name,
                "question": question,
                "question_original": question_original,
                "retrieved_pos": retrieved_pos,
                "context": context,
                "retrieval_hit": any(h["paper_id"] == paper_id for h in retrieved_pos),
            }
        )
        if max_examples is not None and len(examples) >= max_examples:
            break

    hits = sum(1 for e in examples if e["retrieval_hit"])
    print(
        f"E+ Hit@{top_k_pos}: {hits}/{len(examples)} "
        f"({100.0 * hits / max(len(examples), 1):.1f}%)"
    )

    # ---- Stage 1: Policy ----
    print(f"Stage 1 Policy on {len(examples)} examples...")
    policy_prompts = [
        build_policy_prompt(
            e["question"], e["context"], tokenizer, paper_name=e["paper_name"]
        )
        for e in examples
    ]
    policy_raw = llm.generate(policy_prompts, policy_params)
    policy_outs = []
    for i, o in enumerate(policy_raw):
        try:
            policy_outs.append(parse_policy_output(o.outputs[0].text))
        except Exception as exc:
            raise RuntimeError(
                f"Policy parse failed for example {i}: {o.outputs[0].text!r}"
            ) from exc

    # ---- Stage 2: Falsifier ----
    print("Stage 2 Falsifier...")
    falsifier_prompts = [
        build_falsifier_prompt(
            e["paper_name"],
            p["claims"],
            tokenizer,
            question=e["question"],
            answer=p["y"],
        )
        for e, p in zip(examples, policy_outs)
    ]
    falsifier_raw = llm.generate(falsifier_prompts, falsifier_params)
    falsifier_claims = []
    for i, (p, o) in enumerate(zip(policy_outs, falsifier_raw)):
        try:
            fout = parse_falsifier_output(o.outputs[0].text)
        except Exception:
            fout = {"claims": []}
        falsifier_claims.append(_align_falsifier_claims(p["claims"], fout))

    # ---- Stage 3: RAG(E−) ----
    print("Stage 3 RAG counter-evidence...")
    claim_evidence: list[list[dict]] = []
    for e, claims in zip(examples, falsifier_claims):
        per_claim = []
        exclude = e["id"] if exclude_source_paper else None
        for c in claims:
            hits_neg = retrieve_counter_evidence(
                retriever,
                c["anti_queries"],
                top_k=top_k_neg,
                exclude_paper_id=exclude,
            )
            per_claim.append(
                {
                    "claim_id": c["claim_id"],
                    "text": c["text"],
                    "anti_queries": c["anti_queries"],
                    "evidence_neg": hits_neg,
                    "evidence_text": format_retrieved_context(
                        hits_neg, max_chars=MAX_CONTEXT_CHARS
                    ),
                }
            )
        claim_evidence.append(per_claim)

    # ---- Stage 4: Adjudicator ----
    print("Stage 4 Adjudicator...")
    adj_jobs = []  # (ex_idx, claim_idx)
    adj_prompts = []
    for ex_i, (e, p, claims_ev) in enumerate(
        zip(examples, policy_outs, claim_evidence)
    ):
        for c_i, ce in enumerate(claims_ev):
            adj_jobs.append((ex_i, c_i))
            adj_prompts.append(
                build_adjudicator_prompt(
                    {"claim_id": ce["claim_id"], "text": ce["text"]},
                    ce["evidence_text"],
                    tokenizer,
                    question=e["question"],
                    answer=p["y"],
                )
            )

    adj_raw = llm.generate(adj_prompts, adjudicator_params) if adj_prompts else []
    verdicts_by_ex: list[list[dict]] = [[] for _ in examples]
    for (ex_i, c_i), o in zip(adj_jobs, adj_raw):
        ce = claim_evidence[ex_i][c_i]
        verdict = parse_adjudicator_output(
            o.outputs[0].text, claim_id=ce["claim_id"]
        )
        verdicts_by_ex[ex_i].append(verdict)

    # ---- Stage 5+6: Reward + audit ----
    print("Stage 5–6 Reward + audit...")
    rollouts = []
    for e, p, claims_ev, verdicts in zip(
        examples, policy_outs, claim_evidence, verdicts_by_ex
    ):
        reward_info = compute_reward(verdicts)
        claims_audit = []
        for ce, v in zip(claims_ev, verdicts):
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

        rollouts.append(
            {
                "id": e["id"],
                "paper_name": e["paper_name"],
                "question_original": e["question_original"],
                "question": e["question"],
                "retrieval_hit_pos": e["retrieval_hit"],
                "retrieved_pos": [
                    {
                        "paper_id": h["paper_id"],
                        "paper_name": h["paper_name"],
                        "section": h["section"],
                        "chunk_id": h["chunk_id"],
                        "score": h["score"],
                        "text": h["text"][:400],
                    }
                    for h in e["retrieved_pos"]
                ],
                "y": p["y"],
                "claims": claims_audit,
                "reward": reward_info["reward"],
                "reward_breakdown": {
                    "n_claims": reward_info["n_claims"],
                    "n_refuted": reward_info["n_refuted"],
                    "n_supported": reward_info["n_supported"],
                    "n_insufficient": reward_info["n_insufficient"],
                    "claim_rewards": reward_info["claim_rewards"],
                },
            }
        )

    save_jsonl(dataset=rollouts, path=outputs_path)
    print(f"Wrote {len(rollouts)} audited rollouts -> {outputs_path}")
    if rollouts:
        r0 = rollouts[0]
        print(f"Sample y: {r0['y'][:160]}")
        print(
            f"Sample reward: {r0['reward']:.3f} "
            f"(refuted={r0['reward_breakdown']['n_refuted']}, "
            f"supported={r0['reward_breakdown']['n_supported']}, "
            f"insufficient={r0['reward_breakdown']['n_insufficient']})"
        )
        for c in r0["claims"]:
            print(f"  [{c['claim_id']}] {c['verdict']}: {c['text'][:80]}")


if __name__ == "__main__":
    main(
        generated_path=str(ROOT / "data" / "train_data" / "generated.jsonl"),
        rag_path=str(ROOT / "data" / "train_data" / "rag.jsonl"),
        outputs_path=str(ROOT / "data" / "train_data" / "rollouts_full_smoke.jsonl"),
        max_examples=3,
        top_k_pos=5,
        top_k_neg=3,
        exclude_source_paper=True,
    )
