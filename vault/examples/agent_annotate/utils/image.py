import base64
from io import BytesIO
from typing import Union

import PIL.Image


def image_to_base64(
    image: Union[bytes, PIL.Image.Image, str], format: str = "PNG", quality: int = 95
) -> str:
    """
    转换图像为 base64

    Args:
        image: bytes / PIL.Image / 文件路径
        format: 图像格式 (PNG, JPEG)
        quality: JPEG 质量 (1-100)

    Returns:
        base64 字符串
    """
    # 已经是 bytes
    if isinstance(image, bytes):
        return base64.b64encode(image).decode("utf-8")

    # PIL.Image
    if isinstance(image, PIL.Image.Image):
        img = image.convert("RGB")
        buf = BytesIO()
        if format.upper() == "JPEG":
            img.save(buf, format=format, quality=quality)
        else:
            img.save(buf, format=format)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # 文件路径
    if isinstance(image, str):
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    raise TypeError(f"Unsupported image type: {type(image)}")


def make_image_message(
    image: Union[bytes, PIL.Image.Image, str],
    format: str = "PNG",
    max_pixels: int | None = None,
) -> dict:
    """
    创建 OpenAI 格式的图像消息

    Args:
        image: 图像输入
        format: 图像格式
        max_pixels: 最大像素数（可选）

    Returns:
        {"type": "image_url", "image_url": {"url": "data:..."}}
    """
    b64 = image_to_base64(image, format=format)
    mime = f"image/{format.lower()}"

    msg = {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}

    if max_pixels is not None:
        msg["max_pixels"] = max_pixels

    return msg
