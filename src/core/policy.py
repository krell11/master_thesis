import json
import re

from transformers import AutoTokenizer

SYSTEM_POLICY_PROMPT = """
                            You answer scientific paper questions using ONLY the provided context.
                            Then split your answer into atomic factual claims.
                            Rules:
                            - Answer the question; do not refuse if context is partial — say what the context supports
                            - Each claim must be a single checkable statement entailed by your answer y
                            - Do not invent facts absent from the context
                            - Prefer concrete names (paper method, dataset, numbers) over "this paper" / "the authors"
                            - Output ONLY valid JSON, no markdown, no extra text
                            JSON schema:
                            {"y": "<final answer string>", "claims": [{"claim_id": "c0", "text": "<atomic claim>"}, ...]}
                            Use 1–5 claims. claim_id = c0, c1, ...
"""


def build_policy_prompt(question: str, paper_name: str, context: str, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_POLICY_PROMPT.strip()},
        {
            "role": "user",
            "content": (
                f"Paper: {paper_name}\n\n"
                f"Question: {question}\n\n"
                f"Context:\n{context}"
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt, str):
        raise TypeError(
            f"apply_chat_template must return str, got {type(prompt)}. "
            "Check tokenize=False / add_generation_prompt spelling."
        )
    return prompt


def parse_policy_output(raw_text: str) -> dict:
    text = raw_text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if "y" not in data:
        raise ValueError(f"Policy JSON missing 'y': {data!r}")

    raw_claims = data.get("claims", [])
    claims = []
    for i, c in enumerate(raw_claims):
        if isinstance(c, str):
            claims.append({"claim_id": f"c{i}", "text": c.strip()})
        elif isinstance(c, dict):
            text_c = c.get("text") or c.get("claim") or c.get("content")
            if not text_c:
                raise ValueError(f"Claim dict missing text: {c!r}")
            claims.append(
                {
                    "claim_id": str(c.get("claim_id") or c.get("id") or f"c{i}"),
                    "text": str(text_c).strip(),
                }
            )
        else:
            raise TypeError(f"Unexpected claim type: {type(c)} {c!r}")
    if not claims:
        # fallback: treat whole answer as one claim
        claims = [{"claim_id": "c0", "text": str(data["y"]).strip()}]
    data["claims"] = claims[:3]
    return data

