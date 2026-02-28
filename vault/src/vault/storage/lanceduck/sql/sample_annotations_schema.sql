-- ===================================================================
-- ==  扩展多模态 Schema 以支持 Sample Annotations 的 SQL 脚本      ==
-- ==  此脚本可以安全地在已有的数据库上重复运行。                 ==
-- ===================================================================

-- 1. 创建核心的 `sample_annotations` 表
-- 作用：作为所有 Sample Annotation 结果的中央仓库，存储具体的标注值和创建者信息。
CREATE TABLE IF NOT EXISTS sample_annotations (
    id UUID PRIMARY KEY,
    name VARCHAR NOT NULL,                      -- Sample Annotation 的名称, e.g., 'clip_score'
    creator_id UUID REFERENCES creators(id),    -- 关联到 creators 表，复用其作为 Sample Annotation 的创建者
    sequence_id UUID,

    -- 存储不同类型结果的列
    value_float FLOAT,                          -- 存储浮点数 Sample Annotation 结果
    value_json JSON                             -- 存储复杂的结构化 Sample Annotation 结果
);

-- 2. 创建 `sample_annotation_elements` 关联表
-- 作用：描述构成一个 Sample Annotation 的具体元素及其扮演的角色，解决多对多关系的歧义问题。
CREATE TABLE IF NOT EXISTS sample_annotation_elements (
    sample_annotation_id UUID NOT NULL REFERENCES sample_annotations(id),
    element_id UUID NOT NULL,               -- 指向 images.id 或 texts.id 等
    element_type VARCHAR NOT NULL,          -- 区分 ID 来源, e.g., 'image', 'text'
    role VARCHAR NOT NULL,                  -- 元素扮演的角色, e.g., 'source_image'

    PRIMARY KEY (sample_annotation_id, element_id, role)
);


-- ===================================================================
-- ==  为新表和现有表添加推荐的索引以优化查询性能                 ==
-- ===================================================================

-- 3. 为 `sample_annotations` 表添加索引
-- 目的：加速按 Sample Annotation 名称和创建者查询。
CREATE INDEX IF NOT EXISTS idx_sample_annotations_name ON sample_annotations(name);
CREATE INDEX IF NOT EXISTS idx_sample_annotations_creator_id ON sample_annotations(creator_id);

-- 4. 为 `sample_annotation_elements` 表添加索引
-- 目的：根据一个或多个元素，高效地反向查找它们参与构成的 Sample Annotation。
CREATE INDEX IF NOT EXISTS idx_sample_annotation_elements_element_id_role ON sample_annotation_elements(element_id, role);