SELECT
    -- 如果 source 是 NULL (总计行)，则显示 'Overall Total'
    COALESCE(source, 'Overall Total') AS source,

    -- 基本统计
    COUNT(id) AS total_texts,

    -- 内容长度统计 (单位: 字符)
    ROUND(AVG(LENGTH(content)), 2) AS avg_content_length,
    MIN(LENGTH(content)) AS min_content_length,
    MAX(LENGTH(content)) AS max_content_length,
    MEDIAN(LENGTH(content)) AS median_content_length,
    ROUND(STDDEV_SAMP(LENGTH(content)), 2) AS stddev_content_length,
FROM
    texts
GROUP BY
    -- 使用 GROUPING SETS 来同时按 source 分组和计算总计
    GROUPING SETS((source), ())
ORDER BY
    -- 将总计行放在结果的最后
    CASE WHEN source IS NULL THEN 1 ELSE 0 END, source;