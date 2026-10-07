from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="cj-mills/hagrid-classification-512p-no-gesture-150k-zip",
    repo_type="dataset",
    local_dir="./hagrid_data"
)
print("Scaricato in:", path)
