WITH SequenceCounts AS (
    -- 第一步：和之前一样，计算每个独立序列的图片和文本数量
    SELECT
        s.id AS sequence_id,
        s.source,
        COUNT(DISTINCT si.image_id) AS image_count,
        COUNT(DISTINCT st.text_id) AS text_count
    FROM
        sequences s
    LEFT JOIN
        sequence_images si ON s.id = si.sequence_id
    LEFT JOIN
        sequence_texts st ON s.id = st.sequence_id
    GROUP BY
        s.id, s.source
)
-- 第二步：使用 GROUPING SETS 进行多级别聚合
SELECT
    -- 当 source 为 NULL 时 (这是总计行)，显示 'Overall Total'
    COALESCE(source, 'Overall Total') AS source,

    -- 序列本身的统计
    COUNT(sequence_id) AS total_sequences,

    -- 每个序列中图片数量的统计
    ROUND(AVG(image_count), 2) AS avg_images_per_sequence,
    SUM(image_count) AS total_images,
    MIN(image_count) AS min_images_per_sequence,
    MAX(image_count) AS max_images_per_sequence,
    MEDIAN(image_count) AS median_images_per_sequence,
    ROUND(STDDEV_SAMP(image_count), 2) AS stddev_images_per_sequence,

    -- 每个序列中文本数量的统计
    ROUND(AVG(text_count), 2) AS avg_texts_per_sequence,
    SUM(text_count) AS total_texts,
    MIN(text_count) AS min_texts_per_sequence,
    MAX(text_count) AS max_texts_per_sequence,
    MEDIAN(text_count) AS median_texts_per_sequence,
    ROUND(STDDEV_SAMP(text_count), 2) AS stddev_texts_per_sequence
FROM
    SequenceCounts
GROUP BY
    -- 指定两个分组集：(1) 按 source 分组 (2) 总计分组 ()
    GROUPING SETS((source), ())
ORDER BY
    -- 将总计行放在最后，其他按 source 字母顺序排列
    CASE WHEN source IS NULL THEN 1 ELSE 0 END, source;