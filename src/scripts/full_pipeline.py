"""Full falsification pipeline (offline):

  x → RAG(E+) → Policy(y, claims) → score_policy_response (Falsifier → RAG(E−) → Adjudicator → r) → audit JSONL
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

from src.core.falsification_score import score_policy_response
from src.core.policy import build_policy_prompt
from src.core.retriever import CorpusRetriever, format_retrieved_context
from src.utils import load_jsonl, save_jsonl

model_path = str(ROOT / "models")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

MAX_CONTEXT_CHARS = 6000
TOP_K_POS = 5
TOP_K_NEG = 3


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

    def make_generate(params: SamplingParams):
        def _generate(prompts: list[str]) -> list[str]:
            outs = llm.generate(prompts, params)
            return [o.outputs[0].text for o in outs]

        return _generate

    falsifier_generate = make_generate(falsifier_params)
    adjudicator_generate = make_generate(adjudicator_params)

    examples = []
    for row in generated:
        paper_id = row["id"]
        paper_name = row.get("title") or row.get("paper_name") or ""
        question = (row.get("qas") or row.get("question") or "").strip()
        question_original = (row.get("question") or question).strip()
        if not question:
            continue

        retrieved_pos = retriever.retrieve(question, top_k=top_k_pos)
        context = format_retrieved_context(
            retrieved_pos, max_chars=MAX_CONTEXT_CHARS
        )
        examples.append(
            {
                "id": paper_id,
                "paper_name": paper_name,
                "question": question,
                "question_original": question_original,
                "retrieved_pos": retrieved_pos,
                "context": context,
                "retrieval_hit": any(
                    h["paper_id"] == paper_id for h in retrieved_pos
                ),
            }
        )
        if max_examples is not None and len(examples) >= max_examples:
            break

    hits = sum(1 for e in examples if e["retrieval_hit"])
    print(
        f"E+ Hit@{top_k_pos}: {hits}/{len(examples)} "
        f"({100.0 * hits / max(len(examples), 1):.1f}%)"
    )

    print(f"Stage 1 Policy on {len(examples)} examples...")
    policy_prompts = [
        build_policy_prompt(
            e["question"], e["context"], tokenizer, paper_name=e["paper_name"]
        )
        for e in examples
    ]
    policy_raw = llm.generate(policy_prompts, policy_params)

    print("Stages 2–6 Falsifier → E− → Adjudicator → reward...")
    rollouts = []
    for e, o in zip(examples, policy_raw):
        solution_str = o.outputs[0].text
        scored = score_policy_response(
            question=e["question"],
            paper_name=e["paper_name"],
            paper_id=e["id"],
            solution_str=solution_str,
            retriever=retriever,
            tokenizer=tokenizer,
            falsifier_generate=falsifier_generate,
            adjudicator_generate=adjudicator_generate,
            top_k_neg=top_k_neg,
            exclude_source_paper=exclude_source_paper,
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
                "y": scored.get("y"),
                "claims": scored.get("claims", []),
                "parse_ok": scored.get("parse_ok", False),
                "reward": scored["reward"],
                "reward_breakdown": scored["reward_breakdown"],
            }
        )

    save_jsonl(dataset=rollouts, path=outputs_path)
    print(f"Wrote {len(rollouts)} audited rollouts -> {outputs_path}")
    if rollouts:
        r0 = rollouts[0]
        print(f"Sample y: {(r0.get('y') or '')[:160]}")
        print(
            f"Sample reward: {r0['reward']:.3f} "
            f"(refuted={r0['reward_breakdown']['n_refuted']}, "
            f"supported={r0['reward_breakdown']['n_supported']}, "
            f"insufficient={r0['reward_breakdown']['n_insufficient']})"
        )
        for c in r0.get("claims") or []:
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
