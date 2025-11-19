from huggingface_hub import snapshot_download

snapshot_download(repo_id="openai-community/gpt2", repo_type="model", local_dir="E:\Downloads", resume_download=True)