import json
import re

from transformers import AutoTokenizer

SYSTEM_POLICY_PROMPT = """
You answer scientific paper questions using ONLY the retrieved context passages.
Then split your answer into atomic factual claims.

Rules:
- Use only facts supported by the retrieved passages
- If evidence is missing or conflicting, say what the passages support
- Prefer concrete names (paper/method/dataset/numbers) over "this paper" / "the authors"
- Each claim must be a single checkable statement entailed by your answer y
- Output ONLY valid JSON, no markdown, no extra text

JSON schema:
{"y": "<final answer string>", "claims": [{"claim_id": "c0", "text": "<atomic claim>"}, ...]}
Use 1–5 claims. claim_id = c0, c1, ...
"""


def build_policy_prompt(
    question: str,
    context: str,
    tokenizer: AutoTokenizer,
    paper_name: str | None = None,
) -> str:
    header = f"Target paper (optional hint): {paper_name}\n\n" if paper_name else ""
    messages = [
        {"role": "system", "content": SYSTEM_POLICY_PROMPT.strip()},
        {
            "role": "user",
            "content": (
                f"{header}"
                f"Question: {question}\n\n"
                f"Retrieved context:\n{context}"
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


def _fix_invalid_json_escapes(s: str) -> str:
    """Escape backslashes that are not valid JSON escape sequences (e.g. LaTeX \\mathcal)."""
    out: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt in '"\\/bfnrt':
                out.append(s[i : i + 2])
                i += 2
                continue
            if (
                nxt == "u"
                and i + 5 < len(s)
                and all(c in "0123456789abcdefABCDEF" for c in s[i + 2 : i + 6])
            ):
                out.append(s[i : i + 6])
                i += 6
                continue
            out.append("\\\\")
            out.append(nxt)
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def parse_policy_output(raw_text: str) -> dict:
    text = raw_text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = json.loads(_fix_invalid_json_escapes(text))
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
        claims = [{"claim_id": "c0", "text": str(data["y"]).strip()}]
    data["claims"] = claims[:3]
    return data
