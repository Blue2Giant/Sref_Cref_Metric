CREATE TABLE IF NOT EXISTS creators (
    id UUID PRIMARY KEY,
    name VARCHAR NOT NULL,
    meta JSON
);

CREATE TABLE IF NOT EXISTS annotations (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    type VARCHAR,
    creator_id UUID REFERENCES creators(id),
    meta JSON
);

CREATE TABLE IF NOT EXISTS images (
    id UUID PRIMARY KEY,
    uri VARCHAR NOT NULL,
    source VARCHAR,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS texts (
    id UUID PRIMARY KEY,
    content TEXT NOT NULL,
    uri VARCHAR,
    source VARCHAR,
    language VARCHAR(10),
);

CREATE TABLE IF NOT EXISTS sequences (
    id UUID PRIMARY KEY,
    uri VARCHAR,
    source VARCHAR,
    meta JSON
);


-- 图片与 annotation 的关联表
CREATE TABLE IF NOT EXISTS image_annotations (
    image_id UUID NOT NULL REFERENCES images(id),
    annotation_id UUID NOT NULL REFERENCES annotations(id),
    PRIMARY KEY (image_id, annotation_id)
);

-- 文本与 annotation 的关联表
CREATE TABLE IF NOT EXISTS text_annotations (
    text_id UUID NOT NULL REFERENCES texts(id),
    annotation_id UUID NOT NULL REFERENCES annotations(id),
    PRIMARY KEY (text_id, annotation_id)
);

-- 序列与图片的关联表
-- 存储图片在序列中的位置
CREATE TABLE IF NOT EXISTS sequence_images (
    sequence_id UUID NOT NULL REFERENCES sequences(id),
    image_id UUID NOT NULL REFERENCES images(id),
    "index" VARCHAR NOT NULL,  -- 图片在序列中的索引/顺序
    PRIMARY KEY (sequence_id, image_id, "index")
);

-- 序列与文本的关联表
-- 存储文本在序列中的位置
CREATE TABLE IF NOT EXISTS sequence_texts (
    sequence_id UUID NOT NULL REFERENCES sequences(id),
    text_id UUID NOT NULL REFERENCES texts(id),
    "index" VARCHAR NOT NULL,   -- 文本在序列中的索引/顺序
    PRIMARY KEY (sequence_id, text_id, "index")
);


-- 加速通过 annotation_id 查找 image
CREATE INDEX IF NOT EXISTS idx_image_annotations_annotation_id ON image_annotations(annotation_id);

-- 加速通过 annotation_id 查找 text
CREATE INDEX IF NOT EXISTS idx_text_annotations_annotation_id ON text_annotations(annotation_id);

-- 加速标注名称和类型查找
CREATE INDEX IF NOT EXISTS idx_annotations_name ON annotations(name);
CREATE INDEX IF NOT EXISTS idx_annotations_type ON annotations(type);

-- 加速图片和文本的 URI/Source 查找 (如果这些列是唯一的，可使用 UNIQUE 索引)
CREATE INDEX IF NOT EXISTS idx_images_uri ON images(uri);
CREATE INDEX IF NOT EXISTS idx_texts_uri ON texts(uri);
CREATE INDEX IF NOT EXISTS idx_images_source ON images(source);
CREATE INDEX IF NOT EXISTS idx_texts_source ON texts(source);

-- 加速从 creator 查找 annotations
CREATE INDEX IF NOT EXISTS idx_annotations_creator_id ON annotations(creator_id);

-- 加速通过 image_id 反向查找 sequence
CREATE INDEX IF NOT EXISTS idx_sequence_images_image_id ON sequence_images(image_id);
-- 加速通过 text_id 反向查找 sequence
CREATE INDEX IF NOT EXISTS idx_sequence_texts_text_id ON sequence_texts(text_id);