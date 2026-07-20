import json
import os
from pathlib import Path

# No CUDA toolkit (nvcc) in WSL — avoid FlashInfer JIT that needs /usr/local/cuda
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
model_path = str(ROOT / "models")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)



def process_dataset(dataset: list, outputs_path: str):
    from vllm import LLM, SamplingParams
    sampling_params = SamplingParams(n = 1, top_p=1, temperature = 0.3, seed = 999, max_tokens = 512)
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        max_num_seqs=16,
        language_model_only=True,
        trust_remote_code=True,
    )
    outputs = llm.generate(dataset, sampling_params)
    with open(outputs_path, "w", encoding="utf-8") as f:
        for output in outputs:
            generated_sample= output.outputs[0].text
            f.write(json.dumps({"qas":generated_sample}) + "\n")


def load_rag_by_id(idx, rag_path: str, id_to_ensure) -> dict[str, dict]:
    papers = ""
    with open(rag_path, encoding="utf-8") as f:
        for idx_line, line in enumerate(f):
            if idx_line == idx:
                line = line.strip()
                row = json.loads(line)
                if row["id"] == id_to_ensure:
                    papers = row["paragraphs"][0]
    return papers


def create_dataset(dataset_path: str, rag_dataset: str):
    prompts = []
    with open(dataset_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            sample = json.loads(line)
            paper_name = sample["title"]
            paper_context = load_rag_by_id(idx, rag_dataset,  sample["id"])
            for question in sample["qas"]["question"]:
                prompt = f"""
                            You are given a question that adresses a paper named {paper_name}.\n\n
                            Here is the question you need to rephrase: {question} \n\n
                            Here is the paper's summary you are working with, try to make question specific {paper_context}\n\n
                            Return just a question no need to specify the reasoning behind the question you decided to make. It should be possible to understand which paper is question about without reading the paper. 
                            Don't write "context of this paper" specify which paper.
                          """
                messages = tokenizer.apply_chat_template([
                            {"role": "system", "content": "You are a helpful assistant that makes question about scientific papers more specific, your task is to rephrase the question to make it adress the paper name and make it specific so it would be obvious about which paper is the quiestion. it should be possible to understand the source for the question SPECIFY THE PAPER NAME AND ADRESS IT TO MAKE QUESTION SPECIFIC"},
                            {
                                "role": "user",
                                "content": prompt,
                            },
                            ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                
                prompts.append(messages)
    return prompts




if __name__ == "__main__":
    dataset = create_dataset(str(ROOT / "data" / "train_data" / "train.jsonl"), str(ROOT/ "data" / "train_data" / "rag.jsonl"))
    process_dataset(dataset, str(ROOT / "data" / "train_data" / "generated.jsonl"))