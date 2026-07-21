import json

def load_jsonl(path) -> list:
    return [json.loads(sample) for sample in open(path)]

def save_jsonl(dataset, path):
    with open(path, 'w', encoding="utf-8") as f:
        for sample in dataset:
             f.write(json.dumps(sample, ensure_ascii=False) + "\n")