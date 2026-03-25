from huggingface_hub import delete_file

delete_file(
    path_in_repo="train.tar",   # 仓库里的路径
    repo_id="YaxuanLi/UAVM_2026_test",
    repo_type="dataset",
    commit_message="Delete uploaded file"
)