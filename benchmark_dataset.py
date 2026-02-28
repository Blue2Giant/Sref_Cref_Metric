"""
upload benchmark data
"""
from huggingface_hub import HfApi
api = HfApi()
# api.create_repo("Blue2Giant/FreeStyle_Benchmark_800", repo_type="dataset",
# token="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")
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
api.upload_folder(
   folder_path="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_new",
   repo_id="Blue2Giant/FreeStyle_Benchmark_800",
   path_in_repo="bench/cref_sref",
   repo_type="dataset",
)