from __future__ import annotations

import json
import re

from transformers import AutoTokenizer

SYSTEM_ADJUDICATOR_PROMPT = """
You adjudicate whether retrieved evidence CONFLICTS with a claim.

Given one claim and evidence passages (retrieved to try to refute it), output ONE verdict:
- "refuted": evidence clearly contradicts the claim (different number, opposite finding, incompatible fact)
- "supported": evidence agrees with / restates the claim (no contradiction found)
- "insufficient": evidence is unrelated, vague, or too weak to judge

Rules:
- Judge ONLY from the provided evidence passages
- Prefer "insufficient" over guessing
- Do NOT answer the original user question
- Output ONLY valid JSON, no markdown

JSON schema:
{"claim_id": "c0", "verdict": "refuted|supported|insufficient", "rationale": "<one short sentence>"}
""".strip()


def build_adjudicator_prompt(
    claim: dict,
    evidence: str,
    tokenizer: AutoTokenizer,
    question: str | None = None,
    answer: str | None = None,
) -> str:
    header_parts = []
    if question:
        header_parts.append(f"Original question: {question.strip()}")
    if answer:
        header_parts.append(f"Answer being checked: {answer.strip()}")
    header = ("\n".join(header_parts) + "\n\n") if header_parts else ""

    user = (
        f"{header}"
        f"Claim ({claim.get('claim_id', 'c0')}): {claim['text']}\n\n"
        f"Evidence passages:\n{evidence if evidence.strip() else '(no evidence retrieved)'}\n\n"
        "Decide the verdict for this claim."
    )
    messages = [
        {"role": "system", "content": SYSTEM_ADJUDICATOR_PROMPT},
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


def parse_adjudicator_output(raw_text: str, claim_id: str | None = None) -> dict:
    text = raw_text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None

    if not isinstance(data, dict):
        return {
            "claim_id": claim_id or "c0",
            "verdict": "insufficient",
            "rationale": "unparseable adjudicator output",
        }

    verdict = str(data.get("verdict", "insufficient")).strip().lower()
    if verdict not in {"refuted", "supported", "insufficient"}:
        # common aliases
        if "refut" in verdict or "contradict" in verdict:
            verdict = "refuted"
        elif "support" in verdict or "confirm" in verdict:
            verdict = "supported"
        else:
            verdict = "insufficient"

    return {
        "claim_id": str(data.get("claim_id") or claim_id or "c0"),
        "verdict": verdict,
        "rationale": str(data.get("rationale") or data.get("reason") or "").strip(),
    }
