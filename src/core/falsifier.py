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
                            - Every query MUST be searchable: include the paper short name, method, or dataset from the claim/paper
                            - FORBIDDEN vague words: this work, this paper, the authors, they, their method
                            - 1–3 anti_queries per claim
                            - Output ONLY valid JSON, no markdown

                            JSON schema:
                            {
                            "claims": [
                                {
                                "claim_id": "c0",
                                "text": "<same claim text>",
                                "anti_queries": ["<query1>", "<query2>"]
                                }
                            ]
                            }
                            Preserve claim_id order from the input.                        
"""


def build_falsifier_prompt(paper_name, claims, tokenizer: AutoTokenizer) -> str:
    claims_block = "\n".join(
        f"- {c['claim_id']}: {c['text']}" for c in claims
    )
    user = (
        f"Paper: {paper_name}\n\n"
        f"Claims:\n{claims_block}\n\n"
        "Write anti_queries for each claim."
    )

    messages = [
        {"role": "system", "content": SYSTEM_FALSIFIER_PROMPT.strip()},
        {"role": "user", "content": user},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def parse_falsifier_output(raw_text: str) -> dict:
    text = raw_text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    # strip optional ```json fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)
    