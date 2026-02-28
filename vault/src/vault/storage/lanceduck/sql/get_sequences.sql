-- 这个查询用于根据一个或多个 sequence_id 高效地获取所有相关的图片和文本信息。
-- 它利用 DuckDB 的 list() 聚合函数和 struct 功能，为每个序列ID返回一行结果，
-- 其中包含一个图片对象列表和一个文本对象列表。

SELECT
    s.id AS sequence_id,

    -- 聚合所有与该序列关联的图片信息。
    -- [修正] 使用 list(DISTINCT ...) 来防止因 JOIN 产生的重复项。
    -- 当一个 sequence 关联了多张图片和多段文本时，JOIN 会产生笛卡尔积，
    -- 如果不使用 DISTINCT，图片和文本都会在列表中重复出现。
    list(DISTINCT {
        'id': i.id,
        'uri': i.uri,
        'source': i.source,
        'width': i.width,
        'height': i.height,
        'index': si."index"
    }) FILTER (WHERE i.id IS NOT NULL) AS images,

    -- 同样地，聚合所有与该序列关联的文本信息。
    -- [修正] 使用 list(DISTINCT ...) 来防止文本信息重复。
    list(DISTINCT {
        'id': t.id,
        'content': t.content,
        'uri': t.uri,
        'source': t.source,
        'language': t.language,
        'index': st."index"
    }) FILTER (WHERE t.id IS NOT NULL) AS texts

FROM
    sequences AS s
    -- 使用 LEFT JOIN 来确保即使序列只有图片或只有文本（或都没有），它仍然会出现在结果中。
    -- 如果用 INNER JOIN，那么没有图片或没有文本的序列将被过滤掉。
    LEFT JOIN sequence_images AS si ON s.id = si.sequence_id
    LEFT JOIN images AS i ON si.image_id = i.id
    LEFT JOIN sequence_texts AS st ON s.id = st.sequence_id
    LEFT JOIN texts AS t ON st.text_id = t.id
WHERE
    -- 使用 IN 子句来一次性查询多个 sequence_id。
    -- 在实际使用中，('seq_id_1', 'seq_id_2', ...) 会被具体的UUID列表替换。
    s.id IN ?
GROUP BY
    -- 按 sequence_id 分组，以便 list() 函数为每个序列聚合其对应的图片和文本。
    s.id;

