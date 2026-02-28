from vault.schema import ID
from vault.storage.lanceduck.multimodal import MultiModalStorager

storager = MultiModalStorager("/mnt/marmot/chengwei/vault/MultiCref_edit_Echo_data_gemini_watermark_v2")

seq_id_to_meta = {
    ID.from_(item["id"]): item["meta"]
    for item in storager.meta_handler.query_batch('SELECT id, meta FROM "sequences";')
}


for seq_id in seq_id_to_meta:
    sequence_meta = storager.get_sequence_metas([seq_id])
    image_id = sequence_meta[0]["images"][0]["id"]
    image_bytes = storager.get_image_bytes_by_ids([image_id])[image_id]
    print(image_id, type(image_bytes), len(image_bytes))
    print(seq_id_to_meta[seq_id])

    break