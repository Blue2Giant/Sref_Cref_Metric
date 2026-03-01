from vault.schema import ID
from vault.storage.lanceduck.multimodal import MultiModalStorager

storager = MultiModalStorager("/mnt/marmot/chengwei/vault/cref_sref_oneig_filter_part1")

seq_id_to_meta = {
    ID.from_(item["id"]): item["meta"]
    for item in storager.meta_handler.query_batch('SELECT id, meta FROM "sequences";')
}


for seq_id in seq_id_to_meta:
    sequence_meta = storager.get_sequence_metas([seq_id])
    for idx,seq in enumerate(sequence_meta):
        print('index:',idx, "  ",seq.keys())#dict_keys(['sequence_id', 'images', 'texts'])
        for jdx,img in enumerate(seq["images"]):
            print('img_index:',jdx, "  ",img.keys())#dict_keys(['id', 'meta'])
        for kdx,text in enumerate(seq["texts"]):
            print('text_index:',kdx, "  ",text.keys(),'\n',text)#dict_keys(['id', 'meta'])
    image_id = sequence_meta[0]["images"][0]["id"]
    image_bytes = storager.get_image_bytes_by_ids([image_id])[image_id]
    print(image_id, type(image_bytes), len(image_bytes))
    print(seq_id_to_meta[seq_id])

    break