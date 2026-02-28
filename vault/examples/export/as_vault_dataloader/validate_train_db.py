import duckdb
import fire
import tqdm


def validate(db_path: str, batch_size: int = 1000):
    """
    通过迭代方式高效校验 DuckDB 数据库中 'sequences_data' 表的数据完整性。

    Args:
        db_path (str): DuckDB 数据库文件的路径。
        batch_size (int): 每次从数据库中获取并处理的行数。
    """
    con = None
    try:
        # 以只读模式连接到数据库
        con = duckdb.connect(database=db_path, read_only=True)
        print(f"✅ 成功连接到数据库: {db_path}")

        # 首先，获取总行数用于初始化进度条
        try:
            total_rows = con.execute("SELECT COUNT(*) FROM sequences_data").fetchone()[  # pyright: ignore[reportOptionalSubscript]
                0
            ]
        except duckdb.CatalogException:
            print("❌ 错误: 表 'sequences_data' 在数据库中不存在。")
            return

        if total_rows == 0:
            print("ℹ️  表 'sequences_data' 为空，无需校验。")
            return

        print(f"\n🚀 准备校验 {total_rows} 行数据，批次大小: {batch_size}...")

        # 执行查询，准备迭代获取结果
        result = con.execute("SELECT * FROM sequences_data")
        column_names = [desc[0] for desc in result.description]  # pyright: ignore[reportOptionalIterable]

        invalid_row_ids = set()

        # 使用 tqdm 创建进度条
        with tqdm.tqdm(total=total_rows, desc="校验进度", unit="行") as pbar:
            while True:
                # 分批次获取数据
                batch = result.fetchmany(batch_size)
                if not batch:
                    break  # 所有数据都已处理完毕

                # 在当前批次内逐行进行数据校验
                for row_tuple in batch:
                    row = dict(zip(column_names, row_tuple))
                    sequence_id = row.get("sequence_id")
                    vault_path = row.get("vault_path")  # <-- 新增：获取 vault_path
                    is_row_currently_valid = True

                    # 1. 为当前行的 images 和 texts 创建 index 的 Set
                    available_image_indices = {
                        img["index"] for img in row.get("images") or []
                    }
                    available_text_indices = {
                        txt["index"] for txt in row.get("texts") or []
                    }

                    # 2. 遍历 sequence_choices 进行校验
                    sequence_choices = row.get("sequence_choices")
                    if not sequence_choices:
                        continue

                    assert len(sequence_choices) > 0, f"{sequence_id=} {vault_path=}"

                    for choice_list_struct in sequence_choices:
                        if (
                            not choice_list_struct
                            or "choice" not in choice_list_struct
                            or not choice_list_struct["choice"]
                        ):
                            continue

                        for choice in choice_list_struct["choice"]:
                            choice_type = choice.get("type")
                            choice_index = choice.get("index")

                            if (
                                choice_type == "image"
                                and choice_index not in available_image_indices
                            ):
                                # --- 更新错误输出 ---
                                pbar.write(f"\n[校验失败] Sequence ID: {sequence_id}")
                                pbar.write(f"  - Vault Path  : {vault_path}")
                                pbar.write(
                                    f"  - 错误详情    : Image choice 的 index '{choice_index}' 无效。"
                                )
                                pbar.write(
                                    f"  - 可用 Image Indices: {list(available_image_indices) or '[]'}"
                                )
                                is_row_currently_valid = False

                            elif (
                                choice_type == "text"
                                and choice_index not in available_text_indices
                            ):
                                # --- 更新错误输出 ---
                                pbar.write(f"\n[校验失败] Sequence ID: {sequence_id}")
                                pbar.write(f"  - Vault Path  : {vault_path}")
                                pbar.write(
                                    f"  - 错误详情    : Text choice 的 index '{choice_index}' 无效。"
                                )
                                pbar.write(
                                    f"  - 可用 Text Indices: {list(available_text_indices) or '[]'}"
                                )
                                is_row_currently_valid = False

                    if not is_row_currently_valid:
                        invalid_row_ids.add(sequence_id)

                # 更新进度条
                pbar.update(len(batch))

        # 打印最终总结
        print("\n" + "=" * 50)
        print("📊 校验总结:")
        print(f"  - 总共检查行数: {total_rows}")

        invalid_rows_count = len(invalid_row_ids)
        if invalid_rows_count == 0:
            print("  - 发现错误行数: 0")
            print("\n🎉 恭喜！所有数据均满足约束条件！")
        else:
            print(f"  - 发现错误行数: {invalid_rows_count}")
            print("\n❌ 校验发现问题，请检查以上输出的详细信息。")

    except duckdb.Error as e:
        print(f"\n[数据库错误] 操作失败: {e}")
        print("  请确认数据库文件路径是否正确。")
    except Exception as e:
        print(f"\n[未知错误] 脚本运行失败: {e}")
    finally:
        if con:
            con.close()
            print("\n🔌 数据库连接已关闭。")


if __name__ == "__main__":
    fire.Fire(validate)
