import os
import sys
from pathlib import Path

# No CUDA toolkit (nvcc) in WSL — avoid FlashInfer JIT that needs /usr/local/cuda
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.core.policy import build_policy_prompt, parse_policy_output
from src.core.retriever import CorpusRetriever, format_retrieved_context
from src.utils import load_jsonl, save_jsonl

model_path = str(ROOT / "models")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

MAX_CONTEXT_CHARS = 6000
TOP_K = 5


def main(
    generated_path,
    rag_path,
    outputs_path,
    max_examples: int | None = None,
    top_k: int = TOP_K,
):
    rag_dataset = load_jsonl(rag_path)
    generated = load_jsonl(generated_path)

    print(f"Building BM25 index over {len(rag_dataset)} papers...")
    retriever = CorpusRetriever(rag_dataset)
    print(f"Indexed {len(retriever.chunks)} chunks")

    sampling_params = SamplingParams(
        n=1,
        top_p=0.9,
        temperature=0.2,
        seed=999,
        max_tokens=512,
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
        # Prefer rewritten paper-specific query for retrieval + answering
        question = (row.get("qas") or row.get("question") or "").strip()
        question_original = (row.get("question") or question).strip()
        if not question:
            continue

        retrieved = retriever.retrieve(question, top_k=top_k)
        context = format_retrieved_context(retrieved, max_chars=MAX_CONTEXT_CHARS)
        hit = any(h["paper_id"] == paper_id for h in retrieved)
        examples.append(
            {
                "id": paper_id,
                "paper_name": paper_name,
                "question": question,
                "question_original": question_original,
                "retrieved": retrieved,
                "context": context,
                "retrieval_hit": hit,
            }
        )
        if max_examples is not None and len(examples) >= max_examples:
            break

    hits = sum(1 for e in examples if e["retrieval_hit"])
    print(
        f"Retrieval Hit@{top_k}: {hits}/{len(examples)} "
        f"({100.0 * hits / max(len(examples), 1):.1f}%)"
    )
    print(f"Running Policy on {len(examples)} examples (no anti-queries)...")
    print("Sample q0:", examples[0]["question"][:120])
    print("Sample retrieval for q0:")
    for hit in examples[0]["retrieved"][:3]:
        mark = "*" if hit["paper_id"] == examples[0]["id"] else " "
        print(
            f" {mark}[{hit['rank']}] score={hit['score']:.2f} "
            f"{hit['paper_name'][:50]} | {hit['section'][:40]}"
        )

    policy_prompts = [
        build_policy_prompt(
            e["question"],
            e["context"],
            tokenizer,
            paper_name=e["paper_name"],
        )
        for e in examples
    ]
    policy_raw = llm.generate(policy_prompts, sampling_params)

    rollouts = []
    for i, (e, o) in enumerate(zip(examples, policy_raw)):
        raw = o.outputs[0].text
        try:
            p = parse_policy_output(raw)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse policy output for example {i}: {raw!r}"
            ) from exc
        rollouts.append(
            {
                "id": e["id"],
                "paper_name": e["paper_name"],
                "question_original": e["question_original"],
                "question": e["question"],
                "retrieval_hit": e["retrieval_hit"],
                "retrieved": [
                    {
                        "paper_id": h["paper_id"],
                        "paper_name": h["paper_name"],
                        "section": h["section"],
                        "chunk_id": h["chunk_id"],
                        "score": h["score"],
                        "text": h["text"][:500],
                    }
                    for h in e["retrieved"]
                ],
                "y": p["y"],
                "claims": p["claims"],
            }
        )

    save_jsonl(dataset=rollouts, path=outputs_path)
    print(f"Wrote {len(rollouts)} rollouts -> {outputs_path}")
    print("Sample y:", rollouts[0]["y"][:200])


if __name__ == "__main__":
    main(
        generated_path=str(ROOT / "data" / "train_data" / "generated.jsonl"),
        rag_path=str(ROOT / "data" / "train_data" / "rag.jsonl"),
        outputs_path=str(ROOT / "data" / "train_data" / "rollouts_rag_smoke.jsonl"),
        max_examples=3,
        top_k=5,
    )
