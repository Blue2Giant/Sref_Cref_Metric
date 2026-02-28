"""对话管理"""

from dataclasses import dataclass, field
from typing import List

from .image import make_image_message


@dataclass
class Conversation:
    """对话历史管理器"""

    messages: List[dict] = field(default_factory=list)

    def add_system(self, text: str):
        """添加系统消息"""
        self.messages.append(
            {"role": "system", "content": [{"type": "text", "text": text}]}
        )

    def add_user(
        self,
        text: str | None = None,
        images: list | None = None,
        max_pixels: int | None = None,
    ):
        """添加用户消息（文本 + 图像）"""
        content = []

        if images:
            for img in images:
                content.append(make_image_message(img, max_pixels=max_pixels))

        if text:
            content.append({"type": "text", "text": text})

        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, text: str):
        """添加助手回复"""
        self.messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": text}]}
        )

    def get_messages(self) -> list[dict]:
        """导出消息列表"""
        return self.messages.copy()


def build_messages(
    text: str,
    images: list | None = None,
    system: str = "You are a helpful assistant.",
    max_pixels: int | None = None,
) -> list[dict]:
    """
    快速构建单轮消息（工具函数）

    Args:
        text: 用户提示词
        images: 图像列表
        system: 系统提示
        max_pixels: 最大像素数

    Returns:
        OpenAI 消息格式
    """
    conv = Conversation()
    if system:
        conv.add_system(system)
    conv.add_user(text=text, images=images, max_pixels=max_pixels)
    return conv.get_messages()
