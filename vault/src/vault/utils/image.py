import io
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import megfile
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageOps


def to_pil_image(image: bytes | PIL.Image.Image | str | Path) -> PIL.Image.Image:
    if isinstance(image, bytes):
        pil_image = PIL.Image.open(io.BytesIO(image))
    elif isinstance(image, PIL.Image.Image):
        pil_image = image
    elif isinstance(image, str | Path):
        with megfile.smart_open(image, "rb") as f:
            pil_image = PIL.Image.open(f)
    else:
        raise ValueError(f"invalid image type {type(image)}")

    return pil_image


def image_edge_characteristics(
    gray,
    edge_density_weight=0.8,
    edge_continuity_weight=0.2,
    num_top_edges=3,
):
    # 使用Canny边缘检测算法检测图像的边缘
    edges = cv2.Canny(gray, 50, 150)

    # 计算每行和每列的边缘密度
    row_edge_p = edges.mean(axis=1) / 255.0
    col_edge_p = edges.mean(axis=0) / 255.0

    # 计算边缘的连续性
    row_continuity = np.convolve(row_edge_p, [1, -1], mode="valid")
    col_continuity = np.convolve(col_edge_p, [1, -1], mode="valid")

    # 计算综合概率
    edge_density_probability = max(row_edge_p.max(), col_edge_p.max())
    edge_continuity_probability = max(
        np.abs(row_continuity).max(), np.abs(col_continuity).max()
    )

    # 综合考虑边缘密度和边缘连续性
    probability = (
        edge_density_probability * edge_density_weight
        + edge_continuity_probability * edge_continuity_weight
    )

    std_values = []
    # 选择边缘密度最高的几条边
    if row_edge_p.max() > col_edge_p.max():
        top_row_edges = np.argsort(row_edge_p)[-num_top_edges:]
        for row in top_row_edges:
            if row == 0 or row == gray.shape[0] - 1:
                continue
            std_values.append(np.std(gray[0:row, :]))
            std_values.append(np.std(gray[row : gray.shape[0], :]))
    else:
        top_col_edges = np.argsort(col_edge_p)[-num_top_edges:]
        for col in top_col_edges:
            if col == 0 or col == gray.shape[1] - 1:
                continue
            std_values.append(np.std(gray[:, 0:col]))
            std_values.append(np.std(gray[:, col : gray.shape[1]]))

    return float(probability), float(min(std_values))


def image_entropy(gray_np: np.ndarray) -> float:
    """
    用 numpy 实现图像熵,比 PIL.Image.entropy() 快。
    """
    hist = np.bincount(gray_np.ravel(), minlength=256).astype(np.float32)
    hist /= hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def find_pil_font(
    font_path: Optional[str] = None, font_size: int = 14
) -> PIL.ImageFont.FreeTypeFont | PIL.ImageFont.ImageFont:
    """
    尝试找到并加载合适的字体

    Args:
        font_path: 首选字体路径
        font_size: 字体大小

    Returns:
        加载的字体对象,如果找不到则返回默认字体
    """
    # 如果有指定字体路径,优先使用
    if font_path and os.path.exists(font_path):
        try:
            return PIL.ImageFont.truetype(font_path, font_size)
        except IOError:
            pass  # 如果加载失败,继续尝试其他字体

    # 常见字体列表,优先尝试这些字体
    common_fonts = [
        "DejaVuSans.ttf",  # 大多数Linux系统都有
        "Arial.ttf",  # Windows
        "Arial Unicode MS.ttf",  # macOS
        "LiberationSans-Regular.ttf",  # 一些Linux发行版
        "FreeSans.ttf",  # 自由字体
    ]

    # 常见字体路径
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/",
        "/usr/share/fonts/truetype/liberation/",
        "/usr/share/fonts/truetype/freefont/",
        "/usr/share/fonts/truetype/",
        "/usr/local/share/fonts/",
        os.path.expanduser("~/.fonts/"),
    ]

    # 尝试加载字体
    for font_name in common_fonts:
        for path in font_paths:
            try:
                font_path_candidate = os.path.join(path, font_name)
                if os.path.exists(font_path_candidate):
                    return PIL.ImageFont.truetype(font_path_candidate, font_size)
            except (IOError, OSError):
                continue

    # 如果所有字体尝试都失败,使用默认字体
    return PIL.ImageFont.load_default()


def _wrap_text_for_default_font(text: str, max_width: int) -> List[str]:
    """
    为默认字体包装文本(默认字体不支持精确测量)

    Args:
        text: 要包装的文本
        max_width: 最大宽度(像素)

    Returns:
        包装后的文本行列表
    """
    # 默认字体通常是6x10像素
    approx_char_width = 6
    chars_per_line = max_width // approx_char_width
    if chars_per_line <= 0:
        chars_per_line = 1

    # 简单分割文本
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_length = len(word)
        if current_length + word_length + len(current_line) > chars_per_line:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_length
        else:
            current_line.append(word)
            current_length += word_length

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def _wrap_text_for_truetype_font(
    text: str,
    font: PIL.ImageFont.FreeTypeFont | PIL.ImageFont.ImageFont,
    max_width: int,
    draw: PIL.ImageDraw.ImageDraw,
) -> List[str]:
    """
    为TrueType字体包装文本(支持精确测量)

    Args:
        text: 要包装的文本
        font: 字体对象
        max_width: 最大宽度(像素)
        draw: ImageDraw对象用于测量文本

    Returns:
        包装后的文本行列表
    """
    words = text.split()
    lines = []
    current_line = []
    current_width = 0

    for word in words:
        # 在单词前添加空格(除了第一单词)
        if current_line:
            word_with_space = " " + word
        else:
            word_with_space = word

        word_bbox = draw.textbbox((0, 0), word_with_space, font=font)
        word_width = word_bbox[2] - word_bbox[0]

        if current_width + word_width <= max_width:
            current_line.append(word)
            current_width += word_width
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_width = word_width  # 这里不包含空格,因为这是新行的开始

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def _calculate_text_dimensions(
    text_lines: List[str],
    font: PIL.ImageFont.FreeTypeFont | PIL.ImageFont.ImageFont,
    draw: PIL.ImageDraw.ImageDraw,
) -> Tuple[int, int, List[int]]:
    """
    计算文本行集合的总尺寸

    Args:
        text_lines: 文本行列表
        font: 字体对象
        draw: ImageDraw对象用于测量文本

    Returns:
        (总宽度, 总高度, 每行高度列表)
    """
    total_height: int = 0
    max_width: int = 0
    line_heights: list[int] = []

    for line in text_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width: int = int(bbox[2] - bbox[0])
        line_height: int = int(bbox[3] - bbox[1])
        total_height += line_height
        line_heights.append(int(line_height))
        if line_width > max_width:
            max_width = line_width

    return int(max_width), int(total_height), line_heights


def _determine_optimal_font_size(
    text: str,
    font_path: Optional[str],
    max_width: int,
    max_height: int,
    draw: PIL.ImageDraw.ImageDraw,
    wrap_text: bool = True,
) -> Tuple[PIL.ImageFont.FreeTypeFont | PIL.ImageFont.ImageFont, List[str]]:
    """
    确定最佳字体大小以适应可用空间

    Args:
        text: 要显示的文本
        font_path: 字体路径
        max_width: 最大宽度
        max_height: 最大高度
        draw: ImageDraw对象用于测量文本
        wrap_text: 是否启用文本换行

    Returns:
        (最佳字体, 包装后的文本行)
    """
    # 尝试找到一个合适的字体大小
    for test_size in range(24, 8, -1):  # 从24到9尝试
        try:
            test_font = find_pil_font(font_path, test_size)

            # 如果是默认字体,不能调整大小,直接返回
            if test_font == PIL.ImageFont.load_default():
                break

            if wrap_text:
                wrapped_lines = _wrap_text_for_truetype_font(
                    text, test_font, max_width, draw
                )
                _, total_height, _ = _calculate_text_dimensions(
                    wrapped_lines, test_font, draw
                )

                if total_height <= max_height and len(wrapped_lines) > 0:
                    return test_font, wrapped_lines
            else:
                bbox = draw.textbbox((0, 0), text, font=test_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                if text_width <= max_width and text_height <= max_height:
                    return test_font, [text]
        except (AttributeError, IOError):
            # 如果无法调整大小,继续尝试
            continue

    # 如果没有找到合适的字体大小,使用默认设置
    default_font = find_pil_font(font_path, 14)
    if wrap_text:
        if default_font == PIL.ImageFont.load_default():
            wrapped_lines = _wrap_text_for_default_font(text, max_width)
        else:
            wrapped_lines = _wrap_text_for_truetype_font(
                text, default_font, max_width, draw
            )
        return default_font, wrapped_lines
    else:
        return default_font, [text]


def create_text_image(
    text: str,
    size: Tuple[int, int] = (512, 512),
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
    font_path: Optional[str] = None,
    font_size: Optional[int] = None,
    wrap_text: bool = True,
) -> PIL.Image.Image:
    """
    创建包含文本的通知图像

    Args:
        text: 要显示的文本
        size: 图片尺寸 (宽, 高)
        bg_color: 背景颜色 (R, G, B)
        text_color: 文本颜色 (R, G, B)
        font_path: 字体文件路径,如果为None则尝试使用系统字体
        font_size: 字体大小,如果为None则自动计算合适的大小
        wrap_text: 是否自动换行以适应宽度

    Returns:
        PIL.Image.Image: 生成的图片对象
    """
    # 创建图像
    img = PIL.Image.new("RGB", size, color=bg_color)
    draw = PIL.ImageDraw.Draw(img)

    # 确定字体
    if font_size is not None:
        # 使用指定字体大小
        font = find_pil_font(font_path, font_size)
        if wrap_text:
            if font == PIL.ImageFont.load_default():
                text_lines = _wrap_text_for_default_font(text, size[0] - 20)
            else:
                text_lines = _wrap_text_for_truetype_font(
                    text, font, size[0] - 20, draw
                )
        else:
            text_lines = [text]
    else:
        # 自动确定最佳字体大小
        max_width = size[0] - 20  # 留出边距
        max_height = size[1] - 20
        font, text_lines = _determine_optimal_font_size(
            text, font_path, max_width, max_height, draw, wrap_text
        )

    # 计算文本尺寸
    text_width, text_height, line_heights = _calculate_text_dimensions(
        text_lines, font, draw
    )

    # 垂直居中文本
    y = (size[1] - text_height) // 2

    # 绘制每一行文本
    for i, line in enumerate(text_lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        x = (size[0] - line_width) // 2  # 水平居中
        draw.text((x, y), line, fill=text_color, font=font)
        y += line_heights[i]

    return img
