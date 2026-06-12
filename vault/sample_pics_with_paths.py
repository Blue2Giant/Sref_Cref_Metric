from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from vault.schema import ID
from vault.storage.lanceduck.multimodal import MultiModalStorager

VAULT_PATH = "/mnt/marmot/chengwei/vault/cref_sref_oneig_filter_part1"


def _json_default(obj: Any):
    if isinstance(obj, (bytes, bytearray)):
        return f"<bytes len={len(obj)}>"
    return str(obj)


def _print_json(title: str, value: Any):
    print(f"\n=== {title} ===")
    print(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default))


def _extract_path_like_fields(data: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            current = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            if any(token in lowered for token in ("path", "uri", "file", "src")):
                out[current] = value
            out.update(_extract_path_like_fields(value, current))
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            current = f"{prefix}[{idx}]"
            out.update(_extract_path_like_fields(value, current))
    return out


def _query_one_row(storager: MultiModalStorager, table_name: str) -> dict[str, Any]:
    columns = [
        row["column_name"]
        for row in storager.meta_handler.query_batch(f"DESCRIBE {table_name}")
    ]
    select_sql = f'SELECT {", ".join(columns)} FROM {table_name} LIMIT 1'
    rows = storager.meta_handler.query_batch(select_sql)
    return rows[0]


def build_storager(vault_path: str) -> tuple[MultiModalStorager, Path | None]:
    storager = MultiModalStorager(vault_path, read_only=True)
    try:
        storager.meta_handler.query_batch("SELECT 1 AS ok")
        return storager, None
    except Exception as exc:
        print(f"direct metadata open failed: {exc}")

    temp_dir = Path(tempfile.mkdtemp(prefix="vault_metadata_copy_"))
    temp_metadata_path = temp_dir / "metadata.duckdb"
    shutil.copy2(Path(vault_path) / "metadata.duckdb", temp_metadata_path)
    storager = MultiModalStorager(
        vault_path,
        read_only=True,
        metadata_path=str(temp_metadata_path),
    )
    storager.meta_handler.query_batch("SELECT 1 AS ok")
    return storager, temp_dir


def main():
    storager, temp_dir = build_storager(VAULT_PATH)
    try:
        if temp_dir is not None:
            print(f"using copied metadata db: {temp_dir / 'metadata.duckdb'}")

        sequence_row = _query_one_row(storager, "sequences")
        _print_json("first sequence row", sequence_row)
        _print_json(
            "sequence row path-like fields",
            _extract_path_like_fields(sequence_row),
        )

        seq_id = ID.from_(sequence_row["id"])
        sequence_meta = storager.get_sequence_metas([seq_id])[0]
        _print_json("sequence meta keys", list(sequence_meta.keys()))
        _print_json("sequence meta", sequence_meta)

        images = sequence_meta.get("images") or []
        texts = sequence_meta.get("texts") or []
        if not images:
            print("\nno images found in sampled sequence")
            return

        first_image = images[0]
        _print_json("first image entry", first_image)
        _print_json(
            "first image path-like fields",
            _extract_path_like_fields(first_image),
        )

        first_image_id = ID.from_(first_image["id"])
        image_row = storager.meta_handler.query_batch(
            "SELECT * FROM images WHERE id = ? LIMIT 1",
            [first_image_id.to_uuid()],
        )[0]
        _print_json("raw image row", image_row)
        _print_json(
            "raw image row path-like fields",
            _extract_path_like_fields(image_row),
        )

        image_bytes = storager.get_image_bytes_by_ids([first_image_id])[first_image_id]
        print(f"\nimage bytes fetched: {len(image_bytes)} bytes")
        print(f"text items in sampled sequence: {len(texts)}")
    finally:
        storager.meta_handler.close()
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
