import os
import uuid
from pathlib import Path
import megfile
from vault.backend.duckdb import DistributedDuckDBWriter, DuckDBHandler

to_be_merged = [
    "/mnt/sirui/test_vault/train_test4/train.db",
    "/mnt/sirui/test_vault/StepFlow-V2/train.db"
]
merged_dir = "/mnt/sirui/test_vault/test_merge/"


for file_i in to_be_merged:
    megfile.smart_copy(file_i, megfile.smart_path_join(merged_dir, "_duckdb", str(uuid.uuid4().int) + ".duckdb"))

handler = DuckDBHandler(
            megfile.smart_load_text(
                megfile.smart_path_join(
                    Path(__file__).resolve().parent, "sequence_schema.sql"
                )
            ),
        os.path.join(merged_dir, "train.db"),
        read_only=False,
    )

handler.create()
DistributedDuckDBWriter(handler).commit()
