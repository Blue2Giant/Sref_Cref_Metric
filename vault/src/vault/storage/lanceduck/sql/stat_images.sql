SELECT
    -- 如果 source 是 NULL (总计行)，则显示 'Overall Total'
    COALESCE(source, 'Overall Total') AS source,

    -- 基本统计
    COUNT(id) AS total_images,

    -- 宽度统计 (单位: 像素)
    ROUND(AVG(width), 2) AS avg_width,
    MIN(width) AS min_width,
    MAX(width) AS max_width,
    MEDIAN(width) AS median_width,
    ROUND(STDDEV_SAMP(width), 2) AS stddev_width,

    -- 高度统计 (单位: 像素)
    ROUND(AVG(height), 2) AS avg_height,
    MIN(height) AS min_height,
    MAX(height) AS max_height,
    MEDIAN(height) AS median_height,
    ROUND(STDDEV_SAMP(height), 2) AS stddev_height,

    -- 像素面积统计
    ROUND(AVG(width * height), 0) AS avg_pixel_area,
    MIN(width * height) AS min_pixel_area,
    MAX(width * height) AS max_pixel_area,
    MEDIAN(width * height) AS median_pixel_area,

    -- 宽高比统计
    ROUND(AVG(CAST(width AS DOUBLE) / height), 2) AS avg_aspect_ratio,
    ROUND(MEDIAN(CAST(width AS DOUBLE) / height), 2) AS median_aspect_ratio
FROM
    images
WHERE
    height > 0  -- 避免除以零的错误
GROUP BY
    -- 使用 GROUPING SETS 来同时按 source 分组和计算总计
    GROUPING SETS((source), ())
ORDER BY
    -- 将总计行放在结果的最后
    CASE WHEN source IS NULL THEN 1 ELSE 0 END, source;