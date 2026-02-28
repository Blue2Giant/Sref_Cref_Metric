"""OpenAI API 调用封装"""

import json
import re
import time
from typing import Any, Callable, Union

from loguru import logger
from openai import OpenAI


def extract_json(text: str) -> dict:
    """
    从文本中提取 JSON

    支持格式：
    1. 纯 JSON 字符串
    2. ```json ... ```
    3. 文本中嵌入的 {...}
    """
    if not text:
        raise ValueError("响应文本为空")

    # 1. 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 提取 ```json...```
    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. 提取 {...}
    pattern = r"\{.*\}"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法提取 JSON，原始文本: {text[:200]}...")


def call_openai(
    messages: list[dict],
    model: str,
    client: OpenAI,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout: int = 1200,
    is_json: bool = True,
    **extra_params,
) -> Union[dict, str]:
    """
    调用 OpenAI API

    Args:
        messages: 消息列表 [{"role": "user", "content": [...]}]
        model: 模型名称
        client: OpenAI 客户端
        max_tokens: 最大 tokens
        temperature: 温度
        timeout: 超时（秒）
        is_json: 是否解析 JSON
        **extra_params: 额外参数（如 extra_body）

    Returns:
        JSON dict 或 原始文本
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        **extra_params,
    )

    response_text = response.choices[0].message.content
    if response_text is None:
        raise ValueError("API 响应为空")

    return extract_json(response_text) if is_json else response_text


def retry_on_error(
    func: Callable, retry_count: int = 3, retry_delay: float = 2.0, **kwargs
) -> Any:
    """
    重试装饰器

    Args:
        func: 要执行的函数
        retry_count: 重试次数
        retry_delay: 重试延迟（秒）
        **kwargs: 函数参数
    """
    last_error = None

    for attempt in range(retry_count + 1):
        try:
            return func(**kwargs)
        except Exception as e:
            last_error = e
            if attempt < retry_count:
                logger.warning(
                    f"第 {attempt + 1} 次尝试失败，{retry_delay}s 后重试: {e}"
                )
                time.sleep(retry_delay)
            else:
                logger.error(f"所有重试失败: {e}")

    raise last_error  # type: ignore
