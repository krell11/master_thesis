import json
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
        {"role": "system" , "content": SYSTEM_POLICY_PROMPT},
        {"role": "user", "content": f"Paper: {paper_name}\n\nQuestion: {question}\n\nContext:\n{context}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenizer=False, add_generational_prompt=True, enable_thinking=False)


def parse_policy_output(raw_text: str):
    return json.dumps(raw_text, ensure_ascii=False)