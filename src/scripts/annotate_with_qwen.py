import json
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tokenizer =   AutoTokenizer.from_pretrained("../../models/", trust_remote_code=True)
NUM_SEQS: int = 10


def process_dataset(dataset: list, outputs_path: str):
    sampling_params = SamplingParams(n = 1, top_p=1, temperature=0.3, seed=999, max_tokens=512)
    llm = LLM(model="../../models/")
    outputs = llm.generate(dataset, sampling_params)
    with open(outputs_path, "w", encoding="utf-8"):
        for output in outputs:
            generated = output.outputs[0].text
            
            

def create_dataset(dataset_path: str):
    prompts = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            paper_name = sample["title"]
            for question in sample["qas"]:
                prompt = f"""
                            You are given a question that adresses a paper named {paper_name}, your task is to rephrase the question to make it adress the paper name.
                            Here is the question you need to rephrase {question} 
                            Return just a question no need to specify the reasoning behind the question you decided to make.
                          """
                messages = tokenizer.apply_chat_template([
                            {"role": "system", "content": "You are a helpful assistant that makes question about scientific papers more specific"},
                            {
                                "role": "user",
                                "content": prompt,
                            },
                            ], tokenize=False, add_generation_prompt=False)
                
                prompts.append(messages)
    return prompts


if __name__ == "__main__":
    create_dataset("/mnt/e/university/masters/data/train_data/train.jsonl")