"""
Common utility functions for annotation tasks.

This module provides stable, reusable helper functions for:
- Image encoding and format conversion
- API retry logic
- JSON extraction and validation
- Vision-Language Model (VLM) API calls

API Stability: These functions maintain stable interfaces.
"""

import base64
import json
import re
import time
from io import BytesIO
from typing import Any, Callable

import megfile
import PIL.Image
from loguru import logger


def image_to_base64(image: bytes | PIL.Image.Image | str, format="PNG", quality=95):
    """
    Convert image to base64 encoded string.

    Args:
        image: Image input (bytes, PIL.Image, or file path)
        format: Output image format (PNG, JPEG, etc.)
        quality: JPEG quality (1-100), ignored for PNG

    Returns:
        Base64 encoded string
    """
    if isinstance(image, bytes):
        return base64.b64encode(image).decode("utf-8")

    pil_image = None
    image_bytes = None
    if isinstance(image, str):
        with megfile.smart_open(image, "rb") as f:
            pil_image = PIL.Image.open(f).copy()

    if isinstance(image, PIL.Image.Image):
        pil_image = image

    if pil_image is not None:
        pil_image = pil_image.convert("RGB")
        buffered = BytesIO()
        if format.upper() == "JPEG":
            pil_image.save(buffered, format=format, quality=quality)
        else:
            pil_image.save(buffered, format=format)

        image_bytes = buffered.getvalue()
        pil_image.close()
        buffered.close()

    if isinstance(image, bytes):
        image_bytes = image

    assert isinstance(image_bytes, bytes), f"got {type(image_bytes)}"
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return image_base64


def as_image_message(
    image: bytes | PIL.Image.Image | str,
    image_format: str = "PNG",
    min_pixels: int | None = None,
    max_pixels: int | None = None,
):
    """
    Create OpenAI-compatible image message.

    Args:
        image: Image input
        image_format: Image format (PNG, JPEG, etc.)
        min_pixels: Minimum pixel count hint
        max_pixels: Maximum pixel count hint

    Returns:
        Dictionary with OpenAI image_url format
    """
    mime_type = f"image/{image_format.lower()}"
    m = {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{image_to_base64(image, format=image_format)}"
        },
    }
    if min_pixels is not None:
        m["min_pixels"] = min_pixels
    if max_pixels is not None:
        m["max_pixels"] = max_pixels
    return m


def execute_with_retry(
    func: Callable, retry_count: int = 3, retry_delay: float = 2.0, **kwargs
) -> Any:
    """
    Execute function with retry logic.

    Args:
        func: Function to execute
        retry_count: Number of retries on failure
        retry_delay: Delay between retries (seconds)
        **kwargs: Arguments passed to func

    Returns:
        Function execution result

    Raises:
        Exception from last failed attempt
    """
    last_exception: None | BaseException = None

    for attempt in range(retry_count + 1):
        try:
            return func(**kwargs)
        except Exception as e:
            import traceback

            traceback.print_exc()
            last_exception = e
            if attempt < retry_count:
                logger.warning(
                    f"第 {attempt + 1} 次尝试失败, {retry_delay}秒后重试: {e}"
                )
                time.sleep(retry_delay)
            else:
                logger.error(f"所有重试都失败了: {e}")

    assert last_exception is not None
    raise last_exception


def extract_and_validate_json(text: str):
    """
    Extract and validate JSON from response text.

    Handles multiple JSON formats:
    - Direct JSON string
    - JSON wrapped in ```json...```
    - JSON embedded in text (finds {})

    Args:
        text: Raw response text

    Returns:
        Valid JSON

    Raises:
        ValueError: If no valid JSON found
    """
    if not text:
        raise ValueError("响应文本为空")

    # Try parsing text directly as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting ```json...``` blocks
    json_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        json_content = match.group(1).strip()
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"提取的JSON内容无效: {e}")

    # Try finding {} blocks
    json_pattern = r"\{.*\}"
    match = re.search(json_pattern, text, re.DOTALL)

    if match:
        json_content = match.group(0).strip()
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"提取的JSON内容无效: {e}")

    # All methods failed
    text = text.replace("\n", " ").replace("\r", " ")
    raise ValueError(f"无法从响应文本中提取有效的JSON内容。原始文本: {text[:100]}...")


def call_llm(
    text: str,
    model_name: str,
    client,
    max_tokens: int = 8192,
    timeout: int = 1200,
    enable_thinking: bool = False,
):
    messages = [
        {
            "role": "user",
            "content": text,
        },
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,  # pyright: ignore[reportArgumentType]
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
        timeout=timeout,
    )
    response_text = response.choices[0].message.content

    if response_text is None:
        raise ValueError("API响应内容为空")

    return extract_and_validate_json(response_text)


def call_vlm_single(
    image,
    text: str,
    model_name: str,
    client,
    max_tokens: int = 2048,
    timeout: int = 1200,
):
    """
    Call Vision-Language Model with single image.

    Args:
        image: Image input (bytes, PIL.Image, or path)
        text: Text prompt
        model_name: Model identifier
        client: OpenAI client instance
        max_tokens: Maximum response tokens
        timeout: Request timeout (seconds)

    Returns:
        Validated JSON string from model response
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                as_image_message(image, max_pixels=512 * 32 * 32),
                {"type": "text", "text": text},
            ],
        },
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,  # pyright: ignore[reportArgumentType]
        max_tokens=max_tokens,
        extra_body=dict(chat_template_kwargs=dict(add_vision_id=True)),
        timeout=timeout,
    )
    response_text = response.choices[0].message.content

    if response_text is None:
        raise ValueError("API响应内容为空")

    return extract_and_validate_json(response_text)


def call_vlm_compare(
    source_image,
    target_image,
    text: str,
    model_name: str,
    client,
    max_tokens: int = 2048,
    timeout: int = 1200,
):
    """
    Call Vision-Language Model to compare two images.

    Args:
        source_image: First image (before)
        target_image: Second image (after)
        text: Comparison prompt
        model_name: Model identifier
        client: OpenAI client instance
        max_tokens: Maximum response tokens
        timeout: Request timeout (seconds)

    Returns:
        Validated JSON string from model response
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                as_image_message(source_image, max_pixels=512 * 32 * 32),
                as_image_message(target_image, max_pixels=512 * 32 * 32),
                {
                    "type": "text",
                    "text": text,
                },
            ],
        },
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,  # pyright: ignore[reportArgumentType]
        max_tokens=max_tokens,
        extra_body=dict(chat_template_kwargs=dict(add_vision_id=True)),
        timeout=timeout,
    )
    response_text = response.choices[0].message.content

    if response_text is None:
        raise ValueError("API响应内容为空")

    return extract_and_validate_json(response_text)
