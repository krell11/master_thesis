from transformers import AutoTokenizer
import json
import re


SYSTEM_FALSIFIER_PROMPT = """
You are a falsifier for scientific claims.
For each claim, write search queries that would retrieve evidence CONTRADICTING the claim.

Rules:
- Do NOT answer the original user question
- Do NOT write queries that merely restate or support the claim
- Aim at negation, alternative numbers, competing methods/datasets, or conflicting results
- Every query MUST be searchable: include concrete names from the claim/paper
- FORBIDDEN vague words: this work, this paper, the authors, they, their method
- Exactly 2 anti_queries per claim
- Output ONLY one valid JSON object and STOP
- Do not repeat claims

JSON schema example for 2 claims:
{"claims":[{"claim_id":"c0","text":"...","anti_queries":["q1","q2"]},{"claim_id":"c1","text":"...","anti_queries":["q1","q2"]}]}
"""


def build_falsifier_prompt(
    paper_name,
    claims,
    tokenizer: AutoTokenizer,
    question: str | None = None,
    answer: str | None = None,
) -> str:
    claims = claims[:3]
    claims_block = "\n".join(f"- {c['claim_id']}: {c['text']}" for c in claims)
    context_bits = [f"Paper: {paper_name}"]
    if question:
        context_bits.append(f"Question: {question.strip()}")
    if answer:
        context_bits.append(f"Answer y: {answer.strip()}")
    user = (
        "\n".join(context_bits)
        + "\n\n"
        f"Claims ({len(claims)} total — emit exactly these claim_ids once):\n"
        f"{claims_block}\n\n"
        "Write anti_queries for each claim as one JSON object, then stop."
    )

    messages = [
        {"role": "system", "content": SYSTEM_FALSIFIER_PROMPT.strip()},
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


def _unescape(s: str) -> str:
    return json.loads(f'"{s}"')


def parse_falsifier_output(raw_text: str) -> dict:
    text = raw_text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "claims" in data:
            return data
    except json.JSONDecodeError:
        pass

    # Model often emits one flat object repeating claim_id fields.
    # Recover by splitting on each "claim_id" occurrence.
    claims = []
    seen = set()
    parts = re.split(r'(?="claim_id"\s*:)', text)
    for part in parts:
        if '"claim_id"' not in part:
            continue
        cid_m = re.search(r'"claim_id"\s*:\s*"((?:\\.|[^"\\])*)"', part)
        txt_m = re.search(r'"text"\s*:\s*"((?:\\.|[^"\\])*)"', part)
        aq_m = re.search(r'"anti_queries"\s*:\s*\[(.*?)\]', part, re.DOTALL)
        if not (cid_m and txt_m and aq_m):
            continue
        claim_id = _unescape(cid_m.group(1))
        if claim_id in seen:
            continue
        seen.add(claim_id)
        anti_queries = [
            json.loads(q) for q in re.findall(r'"(?:\\.|[^"\\])*"', aq_m.group(1))
        ]
        claims.append(
            {
                "claim_id": claim_id,
                "text": _unescape(txt_m.group(1)),
                "anti_queries": anti_queries,
            }
        )

    if not claims:
        raise json.JSONDecodeError("Could not parse falsifier JSON", text, 0)
    return {"claims": claims}
