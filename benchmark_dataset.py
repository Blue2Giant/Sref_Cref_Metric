"""
upload benchmark data

现在推荐用cli处理，不要用python脚本
三行搞定
brew install huggingface-cli
hf auth login
hf upload Blue2Giant/FreeStyle_Bench /local/path subdir/in/repo --repo-type=dataset
"""
# from huggingface_hub import HfApi
# api = HfApi()
# api.create_repo("Blue2Giant/FreeStyle_Bench", repo_type="dataset",
# token="")
# api.upload_file(
#    path_or_fileobj="/path/to/local/file.csv",
#    path_in_repo="data/file.csv",
#    repo_id="username/my-dataset",
#    repo_type="dataset"
# )

#删除某个路径
# api.delete_file(
#    path_in_repo="data/cref_sref",
#    repo_id="Blue2Giant/Sref_Cref_Benchmark_800",
#    repo_type="dataset",
# )
#上传某个路径
# api.upload_folder(
#    folder_path="/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content",
#    repo_id="Blue2Giant/FreeStyle_Bench",
#    path_in_repo="bench/sref",
#    repo_type="dataset",
# )