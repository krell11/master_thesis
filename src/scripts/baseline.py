from src.utils import load_jsonl, save_jsonl
from src.core.falsifier import build_falsifier_prompt, parse_falsifier_output
from src.core.policy import build_policy_prompt, parse_policy_output

from vllm import SamplingParams, LLM
from transformers import AutoTokenizer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
model_path = str(ROOT / "models")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def main(train_path, rag_path, outputs_path):
    rag_dataset = load_jsonl(rag_path)
    train_dataset = load_jsonl(train_path)

    sampling_params = SamplingParams(n = 1, top_p=1, temperature = 0.3, seed = 999, max_tokens = 512)
    llm  = LLM(model=model_path, gpu_memory_utilization=0.9, max_model_len=4096,
                        max_num_seqs=16, language_model_only=True, trust_remote_code=True)

    examples = []
    for train_row, rag_row in zip(train_dataset, rag_dataset):
        assert train_row["id"] == rag_row["id"]
        context = rag_row["paragraphs"]  
        for q in train_row["qas"]["question"]:
            examples.append({
                            "id": train_row["id"],
                            "paper_name": rag_row.get("paper_name") or train_row["title"],
                            "question": q,                          # один вопрос из train_row["qas"]["question"]
                            "context": context      # текст ЭТОЙ статьи (для Policy)
                            })

    policy_prompts = [build_policy_prompt(e["question"], e["paper_name"], e["context"], tokenizer) for e in examples]
    policy_raw = llm.generate(policy_prompts, sampling_params)
    claims = [parse_policy_output(output) for output in policy_raw]
    falsifier_prompts = [build_falsifier_prompt(e["paper_name"], claims["c"], tokenizer) for e in examples]
    falsifier_raw = llm.generate(falsifier_prompts, sampling_params)
    output = parse_falsifier_output(falsifier_raw)
    save_jsonl(dataset=output, path=outputs_path)
