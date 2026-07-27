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

STOPWORDS = {
    "how",
    "what",
    "when",
    "where",
    "why",
    "which",
    "who",
    "whom",
    "whose",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "and",
    "or",
    "by",
    "with",
    "from",
    "as",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "will",
    "this",
    "that",
    "these",
    "those",
    "their",
    "they",
    "them",
    "its",
    "it",
    "used",
    "using",
    "use",
}

SYSTEM_PROMPT = """
You rewrite ONE question into a paper-specific search query.
Rules:
1. Keep the SAME meaning and key terms as the Original question.
2. MUST include the exact paper title (or Paper short name).
3. Replace vague refs (this paper / this work / the authors / the model) with the paper/method name.
4. Output ONLY the rewritten question. No explanations.
FORBIDDEN: inventing a different topic; copying unrelated example questions.
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


def content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in STOPWORDS}


def preserves_meaning(original: str, rewritten: str, min_recall: float = 0.4) -> bool:
    orig = content_tokens(original)
    if not orig:
        return True
    return len(orig & content_tokens(rewritten)) / len(orig) >= min_recall


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
    # Callable repl — titles may contain backslashes that break template strings.
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

                Paper context (optional hints only; do NOT change the question topic):
                {context_block}

                Original question: {question.strip()}

                Rewrite the Original question so it names this paper.
                Keep the original topic and key terms. Only disambiguate the paper.

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


def finalize_rewrite(
    rewritten: str,
    original: str,
    paper_name: str,
    seen_for_paper: set[str],
) -> str:
    q = (rewritten or "").strip()
    bad = (
        not q
        or not is_paper_specific(q, paper_name)
        or not preserves_meaning(original, q)
        or q.lower() in seen_for_paper
    )
    if bad:
        q = force_paper_specific(original, paper_name)
    # If template still collides (rare), keep it — original questions are unique.
    return q


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
        top_p=1.0,
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

    seen: dict[str, set[str]] = {}
    n_fallback = 0
    with open(outputs_path, "w", encoding="utf-8") as f:
        for record, output in zip(meta, outputs):
            paper_id = record["id"]
            seen.setdefault(paper_id, set())
            raw = clean_generation(output.outputs[0].text)
            rewritten = finalize_rewrite(
                raw, record["question"], record["title"], seen[paper_id]
            )
            if rewritten != raw:
                n_fallback += 1
            seen[paper_id].add(rewritten.lower())
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
    print(f"Wrote {len(meta)} rewrites -> {outputs_path}")
    print(f"Fallbacks to template: {n_fallback}/{len(meta)}")


if __name__ == "__main__":
    prompts, meta = create_dataset(
        str(ROOT / "data" / "train_data" / "train.jsonl"),
        str(ROOT / "data" / "train_data" / "rag.jsonl"),
    )
    print(f"Annotating {len(prompts)} questions...")
    process_dataset(
        prompts,
        meta,
        str(ROOT / "data" / "train_data" / "generated.jsonl"),
    )
