"""Build verl parquet dataset: frozen E+ context + policy chat messages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.policy import build_policy_messages
from src.core.retriever import CorpusRetriever, format_retrieved_context
from src.utils import load_jsonl

DATA_SOURCE = "masters_falsification"
MAX_CONTEXT_CHARS = 6000
TOP_K_POS = 5


def build_rows(
    generated: list[dict],
    retriever: CorpusRetriever,
    *,
    top_k_pos: int = TOP_K_POS,
    max_examples: int | None = None,
    split: str = "train",
) -> list[dict]:
    rows = []
    for idx, row in enumerate(generated):
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
        prompt = build_policy_messages(
            question, context, paper_name=paper_name or None
        )
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": prompt,
                "ability": "paper_qa_falsification",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "id": paper_id,
                    "paper_name": paper_name,
                    "question": question,
                    "question_original": question_original,
                    "retrieval_hit_pos": any(
                        h["paper_id"] == paper_id for h in retrieved_pos
                    ),
                },
            }
        )
        if max_examples is not None and len(rows) >= max_examples:
            break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generated",
        default=str(ROOT / "data" / "train_data" / "generated.jsonl"),
    )
    parser.add_argument(
        "--rag",
        default=str(ROOT / "data" / "train_data" / "rag.jsonl"),
    )
    parser.add_argument(
        "--out_dir",
        default=str(ROOT / "data" / "verl"),
    )
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--top_k_pos", type=int, default=TOP_K_POS)
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="If set, only keep this many rows (smoke).",
    )
    args = parser.parse_args()

    try:
        import datasets
    except ImportError as exc:
        raise SystemExit(
            "Need `datasets` + `pyarrow`. "
            "In conda verl: pip install datasets pyarrow"
        ) from exc

    generated = load_jsonl(args.generated)
    rag = load_jsonl(args.rag)
    print(f"Building BM25 over {len(rag)} papers...")
    retriever = CorpusRetriever(rag)
    print(f"Indexed {len(retriever.chunks)} chunks")

    all_rows = build_rows(
        generated,
        retriever,
        top_k_pos=args.top_k_pos,
        max_examples=args.max_examples,
        split="train",
    )
    n = len(all_rows)
    n_val = max(1, int(n * args.val_ratio)) if n > 1 else 0
    # deterministic tail as val
    train_rows = all_rows[: n - n_val] if n_val else all_rows
    val_rows = all_rows[n - n_val :] if n_val else []
    for r in val_rows:
        r["extra_info"]["split"] = "val"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = datasets.Dataset.from_list(train_rows)
    train_path = out_dir / "train.parquet"
    train_ds.to_parquet(str(train_path))
    print(f"Wrote {len(train_rows)} -> {train_path}")

    if val_rows:
        val_ds = datasets.Dataset.from_list(val_rows)
        val_path = out_dir / "val.parquet"
        val_ds.to_parquet(str(val_path))
        print(f"Wrote {len(val_rows)} -> {val_path}")

    hit = sum(1 for r in all_rows if r["extra_info"]["retrieval_hit_pos"])
    print(f"E+ Hit@{args.top_k_pos}: {hit}/{n} ({100.0 * hit / max(n, 1):.1f}%)")


if __name__ == "__main__":
    main()
