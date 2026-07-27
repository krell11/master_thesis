import json
import os
import re
import sys
from pathlib import Path

# No CUDA toolkit (nvcc) in WSL — avoid FlashInfer JIT that needs /usr/local/cuda
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

model_path = str(ROOT / "models")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

MAX_CONTEXT_CHARS = 3000

GENERIC_PHRASES = (
    "this work",
    "this paper",
    "this model",
    "the authors",
    "their method",
    "the proposed",
)

SYSTEM_PROMPT = """
You rewrite questions into paper-specific search queries.
Output ONLY one question.
MUST include the exact paper title (or its distinctive short name from the title).
Also include a key method/dataset from context when relevant.
FORBIDDEN: this paper, this work, the authors, they, the model (without a name).
""".strip()


def paper_short_name(paper_name: str) -> str:
    return paper_name.split(":")[0].strip() or paper_name


def paper_anchors(paper_name: str) -> list[str]:
    short = paper_short_name(paper_name)
    anchors = {paper_name.lower(), short.lower()}
    for token in re.findall(r"[A-Z][a-z]+(?:[A-Z][a-z]+)+|[A-Z]{2,}", paper_name):
        if len(token) >= 3:
            anchors.add(token.lower())
    return list(anchors)


def is_paper_specific(question: str, paper_name: str) -> bool:
    q = question.lower().strip()
    if len(q) < 12 or "?" not in q:
        return False
    if any(p in q for p in GENERIC_PHRASES):
        return False
    return any(a in q for a in paper_anchors(paper_name))


def force_paper_specific(question: str, paper_name: str) -> str:
    short = paper_short_name(paper_name)
    q = question.strip()
    # Use a callable repl — paper titles may contain backslashes (e.g. \d)
    # which break re.sub when passed as a replacement template string.
    q = re.sub(
        r"\b(this work|this paper|this model|the authors|their method|the proposed)\b",
        lambda _m: short,
        q,
        flags=re.IGNORECASE,
    )
    if not any(a in q.lower() for a in paper_anchors(paper_name)):
        q = q.rstrip("?").rstrip() + f' in "{short}"?'
    if not q.endswith("?"):
        q += "?"
    return re.sub(r"\s+", " ", q).strip()


def clean_generation(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = re.sub(r"^assistant\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    for line in text.splitlines():
        line = line.strip().strip('"').strip("'")
        if line:
            return line
    return text


def build_annotate_prompt(paper_name: str, question: str, context: str) -> str:
    short = paper_short_name(paper_name)
    context_block = context if context else "(No extra paper text available.)"
    user = f"""Paper title: {paper_name}
Paper short name: {short}

Paper context:
{context_block}

Original question: {question.strip()}

Rewrite so a search engine can find THIS paper.

Bad: How is intent annotated?
Bad: How big is the AntiScam dataset?
Good: How is intent annotated in "End-to-End Trainable Non-Collaborative Dialog System" (MISSA)?
Good: How large is the AntiScam dataset in "End-to-End Trainable Non-Collaborative Dialog System"?

Output only the rewritten question."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt, str):
        raise TypeError(f"apply_chat_template must return str, got {type(prompt)}")
    return prompt


def load_rag(rag_path: str) -> list[dict]:
    rows = []
    with open(rag_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def create_dataset(dataset_path: str, rag_path: str) -> tuple[list[str], list[dict]]:
    rag = load_rag(rag_path)
    prompts: list[str] = []
    meta: list[dict] = []

    with open(dataset_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            paper = rag[idx]
            if paper.get("id") and paper["id"] != sample["id"]:
                raise ValueError(
                    f"train/rag mismatch at {idx}: {sample['id']} vs {paper.get('id')}"
                )

            paper_name = paper.get("paper_name") or sample["title"]
            context = (paper.get("text") or "")[:MAX_CONTEXT_CHARS]

            for q_idx, question in enumerate(sample["qas"]["question"]):
                prompts.append(build_annotate_prompt(paper_name, question, context))
                meta.append(
                    {
                        "id": sample["id"],
                        "title": paper_name,
                        "question": question.strip(),
                        "question_index": q_idx,
                    }
                )
    return prompts, meta


def process_dataset(
    prompts: list[str],
    meta: list[dict],
    outputs_path: str,
) -> None:
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        n=1,
        top_p=0.9,
        temperature=0.3,
        seed=999,
        max_tokens=128,
    )
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.75,
        max_model_len=4096,
        max_num_seqs=16,
        language_model_only=True,
        trust_remote_code=True,
    )
    outputs = llm.generate(prompts, sampling_params)

    with open(outputs_path, "w", encoding="utf-8") as f:
        for record, output in zip(meta, outputs):
            rewritten = clean_generation(output.outputs[0].text)
            if not is_paper_specific(rewritten, record["title"]):
                rewritten = force_paper_specific(
                    rewritten or record["question"], record["title"]
                )
            f.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "title": record["title"],
                        "question": record["question"],
                        "qas": rewritten,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    prompts, meta = create_dataset(
        str(ROOT / "data" / "train_data" / "train.jsonl"),
        str(ROOT / "data" / "train_data" / "rag.jsonl"),
    )
    process_dataset(
        prompts,
        meta,
        str(ROOT / "data" / "train_data" / "generated.jsonl"),
    )
