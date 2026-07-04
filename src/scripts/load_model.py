from huggingface_hub import snapshot_download

if __name__ == "__main__":
    repo_id = "Qwen/Qwen3.5-4B" 
    local_dir = "E:\\university\\masters\\models\\" 
    snapshot_download(repo_id=repo_id, local_dir=local_dir)