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
from tqdm import tqdm

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

# from metrics.imagenet_utils import get_embedding_for_prompt, imagenet_templates
def get_embedding_for_prompt(model, prompt):
    # texts = [template.format(prompt) for template in templates]  # format with class
    texts = [prompt]
    # texts = [t.replace('a a', 'a') for t in texts]  # remove double a's
    # texts = [t.replace('the a', 'a') for t in texts]  # remove double a's
    texts = clip.tokenize(texts).cuda()  # tokenize
    class_embeddings = model.encode_text(texts)  # embed with text encoder
    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    class_embedding = class_embeddings.mean(dim=0)
    class_embedding /= class_embedding.norm()
    return class_embedding.float()


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

def load_model(rank):
    logger.info(f"Loading model on rank {rank}")
    device = torch.device(f"cuda:{rank}")
    model_config = AutoConfig.from_pretrained('/data/midjourney/model_zoo/ckpts/dinov2-base')
    # 修改配置
    model_config.output_hidden_states = True
    blipprocessor = AutoImageProcessor.from_pretrained('/data/midjourney/model_zoo/ckpts/dinov2-base')
    blipmodel = AutoModel.from_pretrained('/data/midjourney/model_zoo/ckpts/dinov2-base', config=model_config).to(device)

    blipmodel.eval()
    return blipprocessor, blipmodel,device

def compute_cas(content_image, style_image,blipprocessor, blipmodel,device):
    with torch.no_grad():
        # content_image = content_image.resize((512, 512))
        inputs1 = blipprocessor(images=content_image, return_tensors="pt").to(device)
        outputs1 = blipmodel(**inputs1)
        # print(outputs1)
        image_features1 = outputs1.last_hidden_state
        mean, std = calc_mean_std(image_features1.transpose(-1, -2))
        size = image_features1.transpose(-1, -2).size()
        # content_mean, content_std = calc_mean_std(content_feat)
        normalized_feat = (image_features1.transpose(-1, -2) - mean.expand(
            size)) / std.expand(size)
    

        # style_image = style_image.resize((512, 512))
        with torch.no_grad():
            inputs2 = blipprocessor(images=style_image, return_tensors="pt").to(device)
            outputs2 = blipmodel(**inputs2)
            # print(outputs1)
            image_features2 = outputs2.last_hidden_state
            mean, std = calc_mean_std(image_features2.transpose(-1, -2))
            size = image_features2.transpose(-1, -2).size()
            # content_mean, content_std = calc_mean_std(content_feat)
            normalized_sty = (image_features2.transpose(-1, -2) - mean.expand(
                size)) / std.expand(size)
        cas = torch.mean((normalized_sty - normalized_feat) ** 2, dim=(0, 1, 2)).detach().cpu().item()
        return cas

#定义lance存储格式
def get_schema():
    return pa.schema([
        pa.field("sequence_id", pa.uuid(), nullable=False),
        pa.field("cas", pa.float32(), nullable=False),
    ])


def main(vault_path: str="/mnt/marmot/chengwei/vault/style_transfer_Gemini3_part1",save_path: str="/data/midjourney/xingpeng/vault/vault-label/style_transfer_Gemini3_part1/"):
    torch.distributed.init_process_group(backend='nccl')
    dp_rank = int(os.environ.get("RANK", 0))
    dp_size = int(os.environ.get("WORLD_SIZE", 1))
    logger.info(f"dp_rank: {dp_rank}, dp_size: {dp_size}")
    blipprocessor, blipmodel,device = load_model(dp_rank)

    storager = MultiModalStorager(path=vault_path, read_only=True)
    sequence_ids = storager.meta_handler.query_batch("select id from sequences")
    all_sequence_ids = [ID.from_(s["id"]) for s in sequence_ids]
    # all_sequence_ids = all_sequence_ids[:100]
    all_sequence_ids = all_sequence_ids[dp_rank::dp_size]
    logger.info(f"Total sequences: {len(all_sequence_ids)},each rank will process {len(all_sequence_ids) // dp_size} sequences")
    cas_list = [] 
    #进度条
    for id_index,sequence_id in tqdm(enumerate(all_sequence_ids),desc=f"Processing sequences on rank {dp_rank}"):


        sequence_meta= storager.get_sequence_metas([sequence_id])[0]
        content_image_id = sequence_meta["images"][0]["id"]
        style_image_id = sequence_meta["images"][1]["id"]

        content_image_bytes = storager.get_image_bytes_by_ids([content_image_id])[content_image_id]
        style_image_bytes = storager.get_image_bytes_by_ids([style_image_id])[style_image_id]
        content_image = PIL.Image.open(io.BytesIO(content_image_bytes))
        style_image = PIL.Image.open(io.BytesIO(style_image_bytes))
        
        content_image = content_image.convert("RGB").resize((512, 512))
        style_image = style_image.convert("RGB").resize((512, 512))
        cas = compute_cas(content_image, style_image,blipprocessor, blipmodel,device)
        #将cas存储到lance
        cas_list.append({"sequence_id": sequence_id, "cas": cas})
        #每1w条存储一次
        if len(cas_list) % 10000 == 0:
            table = pa.Table.from_pylist(cas_list, schema=get_schema())
            lance.write_dataset(table, f"{save_path}/cas_sim_{dp_rank}.lance")
            cas_list = []
        #将这两张图拼成一起并保存到本地，图片名称为sequence_id+_cas.png
        # combined_image = PIL.Image.new("RGB", (1024, 512))
        # combined_image.paste(content_image, (0, 0))
        # combined_image.paste(style_image, (512, 0))
        # #按照cas得分 分组保存到不同的文件夹
        # cas_group = int(cas // 0.1)
        # cas_group_dir = f"/data/project/workscript/files/images/cas/cas_sim_{cas_group}"
        # if not os.path.exists(cas_group_dir):
        #     os.makedirs(cas_group_dir)
        # with megfile.smart_open(f"{cas_group_dir}/{sequence_id}_{cas}.png", "wb") as f:
        #     combined_image.save(f, format="PNG")
        logger.info(f"cas: {cas}")

#合并每个dprank的 lance
def merge_lance(dp_size: int,save_path: str="/data/midjourney/xingpeng/vault/vault-label/style_transfer_Gemini3_part1/"):
    for i in range(dp_size):
        table = lance.read_dataset(f"{save_path}/cas_sim_{i}.lance")
        table = table.merge(table, "sequence_id")
        lance.write_dataset(table, f"{save_path}/cas_sim.lance")


    table = lance.read_dataset(f"{save_path}/cas_sim.lance")
    #对sequence_id建立索引
    table.create_scalar_index(
        column="sequence_id",
        index_type="BTREE",
        name="sequence_id_btree_idx",
        replace=True
    )


if __name__ == '__main__':
    import fire
    fire.Fire(main)