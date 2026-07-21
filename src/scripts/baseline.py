import os

# No CUDA toolkit (nvcc) in WSL — avoid FlashInfer JIT that needs /usr/local/cuda
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.core.falsifier import build_falsifier_prompt, parse_falsifier_output
from src.core.policy import build_policy_prompt, parse_policy_output
from src.utils import load_jsonl, save_jsonl

ROOT = Path(__file__).resolve().parents[2]
model_path = str(ROOT / "models")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

MAX_CONTEXT_CHARS = 6000


def main(train_path, rag_path, outputs_path, max_examples: int | None = None):
    rag_dataset = load_jsonl(rag_path)
    train_dataset = load_jsonl(train_path)

    sampling_params = SamplingParams(
        n=1,
        top_p=0.9,
        temperature=0.2,
        seed=999,
        max_tokens=700,
        stop=["\n\n\n"],
    )
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.75,
        max_model_len=8182,
        max_num_seqs=8,
        language_model_only=True,
        trust_remote_code=True,
    )

    examples = []
    for train_row, rag_row in zip(train_dataset, rag_dataset):
        assert train_row["id"] == rag_row["id"], (
            f"train/rag id mismatch: {train_row['id']} vs {rag_row.get('id')}"
        )
        context = (rag_row.get("text") or "")[:MAX_CONTEXT_CHARS]
        paper_name = rag_row.get("paper_name") or train_row["title"]
        for q in train_row["qas"]["question"]:
            examples.append(
                {
                    "id": train_row["id"],
                    "paper_name": paper_name,
                    "question": q.strip(),
                    "context": context,
                }
            )
            if max_examples is not None and len(examples) >= max_examples:
                break
        if max_examples is not None and len(examples) >= max_examples:
            break

    policy_prompts = [build_policy_prompt(e["question"], e["paper_name"], e["context"], tokenizer) for e in examples]
    policy_raw = llm.generate(policy_prompts, sampling_params)
    policy_outs = []
    for i, o in enumerate(policy_raw):
        raw = o.outputs[0].text
        try:
            policy_outs.append(parse_policy_output(raw))
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse policy output for example {i}: {raw!r}"
            ) from e

    print(f"Policy done: {len(policy_outs)} examples")
    print("Sample policy:", policy_outs[0])

    falsifier_prompts = [
        build_falsifier_prompt(e["paper_name"], p["claims"], tokenizer)
        for e, p in zip(examples, policy_outs)
    ]
    falsifier_raw = llm.generate(falsifier_prompts, sampling_params)

    rollouts = []
    for i, (e, p, f) in enumerate(zip(examples, policy_outs, falsifier_raw)):
        raw = f.outputs[0].text
        try:
            fout = parse_falsifier_output(raw)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse falsifier output for example {i}: {raw!r}"
            ) from exc
        rollouts.append(
            {
                "id": e["id"],
                "paper_name": e["paper_name"],
                "question": e["question"],
                "y": p["y"],
                "claims": fout["claims"],
            }
        )

    save_jsonl(dataset=rollouts, path=outputs_path)
    print(f"Wrote {len(rollouts)} rollouts -> {outputs_path}")



if __name__ == "__main__":
    main(
        train_path=str(ROOT / "data" / "train_data" / "train.jsonl"),
        rag_path=str(ROOT / "data" / "train_data" / "rag.jsonl"),
        outputs_path=str(ROOT / "data" / "train_data" / "rollouts_smoke.jsonl"),
        max_examples=3,
    )
