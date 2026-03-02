import json
import sys
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel, AutoConfig
# import clip
import numpy as np
# import pyrallis
import os
import glob
import torch
import argparse
from PIL import Image
from tqdm import tqdm
import shutil
import json
from torchvision import transforms
import torchvision.transforms.functional as F
import ssl
from tqdm import tqdm, trange
import json
import PIL

import glob
import io
import lance
import megfile
from vault.backend.duckdb import DuckDBHandler
from vault.schema import ID
from vault.schema.multimodal import (
    Creator,
    Image,
    MultiModalType,
    PackSequenceIndex,
    SampleAnnotation,
    Text,
)
import megfile
from vault.storage.lanceduck.multimodal import MultiModalStorager
from loguru import logger
import pyarrow as pa
from csd_utils import CSDStyleEmbedding, SEStyleEmbedding

# # from metrics.imagenet_utils import get_embedding_for_prompt, imagenet_templates
# def get_embedding_for_prompt(model, prompt):
#     # texts = [template.format(prompt) for template in templates]  # format with class
#     texts = [prompt]
#     # texts = [t.replace('a a', 'a') for t in texts]  # remove double a's
#     # texts = [t.replace('the a', 'a') for t in texts]  # remove double a's
#     texts = clip.tokenize(texts).cuda()  # tokenize
#     class_embeddings = model.encode_text(texts)  # embed with text encoder
#     class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
#     class_embedding = class_embeddings.mean(dim=0)
#     class_embedding /= class_embedding.norm()
#     return class_embedding.float()


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    # assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def traversal_files(path):
    dirs = []
    files = []
    for item in os.scandir(path):
        if item.is_dir():
            dirs.append(item.path)
        if item.is_file():
            files.append(item.path)
    return dirs, files




#定义lance存储格式
def get_schema():
    return pa.schema([
        pa.field("sequence_id", pa.uuid(), nullable=False),
        pa.field("CSD_embed", pa.list_(pa.float32(), 768)),
        pa.field("SE_embed", pa.list_(pa.float32(), 1280)),
    ])


def main(vault_path: str="/mnt/chengwei/vault/style_transfer_Gemini3_full", save_path: str="/mnt/chengwei/vault/style_transfer_Gemini3_full", machine_id: int=0, machine_size: int=1):
    torch.distributed.init_process_group(backend='nccl')
    dp_rank = int(os.environ.get("RANK", 0))
    dp_size = int(os.environ.get("WORLD_SIZE", 1))
    logger.info(f"dp_rank: {dp_rank}, dp_size: {dp_size}")
    CSD_Encoder = CSDStyleEmbedding(model_path="style_models/checkpoint.pth", device=f"cuda:{dp_rank}")
    SE_Encoder = SEStyleEmbedding(pretrained_path="style_models/models--xingpng--style_encoder", device=f"cuda:{dp_rank}")

    storager = MultiModalStorager(path=vault_path, read_only=True)
    sequence_ids = storager.meta_handler.query_batch("select id from sequences")
    sequence_ids = sorted(sequence_ids, key=lambda x: x["id"])[machine_id::machine_size]
    print(len(sequence_ids))
    all_sequence_ids = [ID.from_(s["id"]) for s in sequence_ids]
    # all_sequence_ids = all_sequence_ids[:100]
    # all_sequence_ids = all_sequence_ids[dp_rank::dp_size]
    logger.info(f"Total sequences: {len(all_sequence_ids)},each rank will process {len(all_sequence_ids) // dp_size} sequences")
    cas_list = [] 
    #进度条
    for id_index,sequence_id in tqdm(enumerate(all_sequence_ids),desc=f"Processing sequences on rank {dp_rank}"):
        if id_index % dp_size != dp_rank:
            continue

        sequence_meta= storager.get_sequence_metas([sequence_id])[0]
        style_image_id = sequence_meta["images"][1]["id"]

        style_image_bytes = storager.get_image_bytes_by_ids([style_image_id])[style_image_id]
        style_image = PIL.Image.open(io.BytesIO(style_image_bytes))
        
        style_image = style_image.convert("RGB").resize((512, 512))
        CSD_embed = CSD_Encoder.get_style_embedding(style_image)
        SE_embed = SE_Encoder.get_style_embedding(style_image)        #将cas存储到lance original_item["id"].to_bytes(),
        cas_list.append({"sequence_id": sequence_id.to_bytes(), "CSD_embed": CSD_embed, "SE_embed": SE_embed})
        #每1w条存储一次
        if len(cas_list) % 10000 == 0:
            table = pa.Table.from_pylist(cas_list, schema=get_schema())
            lance.write_dataset(table, f"{save_path}/style_embedding_{machine_id}_{dp_rank}.lance", mode="append")
            cas_list = []
    if len(cas_list) > 0:
        table = pa.Table.from_pylist(cas_list, schema=get_schema())
        lance.write_dataset(table, f"{save_path}/style_embedding_{machine_id}_{dp_rank}.lance", mode="append")
        #将这两张图拼成一起并保存到本地，图片名称为sequence_id+_cas.png
        # combined_image = PIL.Image.new("RGB", (1024, 512))
        # combined_image.paste(content_image, (0, 0))
        # combined_image.paste(style_image, (512, 0))
        # #按照cas得分 分组保存到不同的文件夹
        # cas_group = int(cas // 0.1)
        # cas_group_dir = f"/data/project/workscript/files/images/cas/style_embedding_{cas_group}"
        # if not os.path.exists(cas_group_dir):
        #     os.makedirs(cas_group_dir)
        # with megfile.smart_open(f"{cas_group_dir}/{sequence_id}_{cas}.png", "wb") as f:
        #     combined_image.save(f, format="PNG")
        # logger.info(f"cas: {cas}")

# 合并每个 dprank 的 lance 文件为一个 lance 文件，修复原有合并逻辑的 bug
def merge_lance(dp_size: int, save_path: str = "/mnt/chengwei/vault/style_transfer_Gemini3_full", world_size: int = 6, output_name: str = "style_embedding.lance"):
    """
    合并 save_path 下所有 style_embedding_{j}_{i}.lance 文件为一个 {output_name}。
    world_size: 默认 6，代表有几个 machine_id（或分组）
    dp_size: 单个 group 内的 rank 数
    """

    output_lance_path = os.path.join(save_path, output_name)

    # 如果存在旧的，先删除
    if os.path.exists(output_lance_path):
        print(f"Removing existing merged lance dataset: {output_lance_path}")
        shutil.rmtree(output_lance_path)

    # 搜集所有需要合并的 lance 文件
    all_lance_files = []
    for j in trange(world_size):
        for i in trange(dp_size):
            path = f"{save_path}/style_embedding_{j}_{i}.lance"
            if os.path.exists(path):
                all_lance_files.append(path)
            else:
                print(f"Warning: {path} does not exist, skipping.")

    if len(all_lance_files) == 0:
        print("No lance files found to merge.")
        return

    # 校验第一个文件获取 schema
    first_ds = lance.dataset(all_lance_files[0])
    schema = first_ds.schema

    with tqdm(total=len(all_lance_files), desc="Merging lance") as pbar:
        for idx, lance_file in enumerate(all_lance_files):
            ds = lance.dataset(lance_file)
            # 分批读取，避免内存爆炸
            for batch in ds.to_batches():
                table = pa.Table.from_batches([batch], schema=schema)
                lance.write_dataset(table, output_lance_path, schema=schema, mode="append" if os.path.exists(output_lance_path) else None)
            pbar.update(1)

    # 建立 BTREE 索引
    ds_merged = lance.dataset(output_lance_path)
    print("Creating BTREE index on 'sequence_id'...")
    ds_merged.create_scalar_index(
        column="sequence_id",
        index_type="BTREE",
        name="sequence_id_btree_idx",
        replace=True
    )
    print("Lance merge and index creation done.")



if __name__ == '__main__':
    import fire
    fire.Fire(merge_lance)