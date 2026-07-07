import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


model_path  = '/workspace/models/'
tokenizer =   AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)



def process_dataset(dataset: list, outputs_path: str):
    sampling_params = SamplingParams(n = 1, top_p=1, temperature = 0.3, seed = 999, max_tokens = 128)
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        max_num_seqs=32,
        language_model_only=True,
        trust_remote_code=True,
    )
    outputs = llm.generate(dataset, sampling_params)
    with open(outputs_path, "w", encoding="utf-8") as f:
        for output in outputs:
            generated_sample= output.outputs[0].text
            f.write(json.dumps({"qas":generated_sample}) + "\n")
            

def create_dataset(dataset_path: str):
    prompts = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            paper_name = sample["title"]
            for question in sample["qas"]["question"]:
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
                            ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                
                prompts.append(messages)
    return prompts


if __name__ == "__main__":
    dataset = create_dataset("/workspace/data/train_data/train.jsonl")
    process_dataset(dataset,"/workspace/data/train_data/generated.jsonl")