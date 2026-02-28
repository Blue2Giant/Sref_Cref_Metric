#!/usr/bin/env python3

import io
import os
import random
import sys
import threading
import time
import traceback
from typing import Any, List, Optional, Tuple
from urllib.parse import urlencode

import gradio as gr
import gradio.themes
import pandas as pd
from loguru import logger
from PIL import Image

from vault.backend.lance import LanceTaker
from vault.schema import ID
from vault.storage.lanceduck.multimodal import MultiModalStorager


class VaultBrowser:
    """Vault数据浏览器类"""

    def __init__(self):
        self.storager: Optional[MultiModalStorager] = None
        self.lance_taker = LanceTaker()
        self.current_sources: List[str] = []
        self.current_sequence_ids: List[str] = []
        self.current_sequence_items: List[dict] = []  # 存储当前序列的所有元素
        self.selected_elements: List[dict] = []  # 存储用户选择的元素

        # 添加线程锁来保护DuckDB操作
        self._duckdb_lock = threading.Lock()
        self._max_retries = 3
        self._retry_delay = 0.1  # 100ms

    def _safe_duckdb_operation(self, operation_func, *args, **kwargs):
        """安全执行DuckDB操作，带重试机制和线程锁"""
        for attempt in range(self._max_retries):
            try:
                with self._duckdb_lock:
                    return operation_func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                # 检查是否是DuckDB内部错误
                if (
                    "INTERNAL Error" in error_msg
                    or "unique_ptr that is NULL" in error_msg
                ):
                    if attempt < self._max_retries - 1:
                        logger.warning(
                            f"DuckDB操作失败，第{attempt + 1}次重试: {error_msg}"
                        )
                        time.sleep(self._retry_delay * (attempt + 1))  # 指数退避
                        continue
                    else:
                        logger.error(
                            f"DuckDB操作最终失败，已重试{self._max_retries}次: {error_msg}"
                        )
                        raise
                else:
                    # 非DuckDB内部错误，直接抛出
                    raise

    @staticmethod
    def generate_notice_image(
        text="[vault] image not found",
        size=(400, 200),
        bg_color=(255, 255, 255),
        text_color=(0, 0, 0),
    ) -> Image.Image:
        from PIL import ImageDraw, ImageFont

        img = Image.new("RGB", size, color=bg_color)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        # 使用 textbbox 计算文本边界
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 居中
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2

        draw.text((x, y), text, fill=text_color, font=font)
        return img

    def load_vault(self, vault_path: str) -> Tuple[str, List[str]]:
        """加载vault数据并返回可用的sources"""
        logger.info(f"开始加载vault: {vault_path}")
        try:
            if not os.path.exists(vault_path):
                logger.error(f"Vault路径不存在: {vault_path}")
                return f"错误: 路径 {vault_path} 不存在", []

            logger.debug(f"创建MultiModalStorager实例，路径: {vault_path}")
            self.storager = MultiModalStorager(vault_path, read_only=True)

            # 使用安全操作包装器查询所有可用的sources
            def query_sources():
                if self.storager is None:
                    logger.warning("Storager为None，返回空列表")
                    return []
                logger.debug(
                    "执行DuckDB查询: SELECT DISTINCT source FROM sequences ORDER BY source"
                )
                return self.storager.meta_handler.query_batch(
                    "SELECT DISTINCT source FROM sequences ORDER BY source"
                )

            logger.debug("开始执行DuckDB查询操作")
            sources = self._safe_duckdb_operation(query_sources)

            if sources is None:
                logger.warning("DuckDB查询返回None，使用空列表")
                sources = []
            source_list = [s["source"] for s in sources if s and s["source"]]
            self.current_sources = source_list

            logger.info(
                f"Vault加载完成，找到 {len(source_list)} 个数据源: {source_list}"
            )

            if not source_list:
                logger.warning("未找到任何数据源")
                return "成功加载vault, 但未找到任何数据源", []

            return f"成功加载vault, 找到 {len(source_list)} 个数据源", source_list

        except Exception as e:
            logger.error(f"加载vault时出错: {str(e)}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return f"加载vault时出错: {str(e)}", []

    def load_sequences(self, selected_sources: List[str]) -> Tuple[str, int, int]:
        """根据选中的sources加载序列ID"""
        logger.info(f"开始加载序列，选中的数据源: {selected_sources}")

        if not self.storager or not selected_sources:
            logger.warning("Storager未初始化或未选择数据源")
            return "请先选择数据源", 0, 0

        try:
            # 使用安全操作包装器获取序列ID
            def get_sequence_ids():
                if self.storager is None:
                    logger.warning("Storager为None，返回空列表")
                    return []
                logger.debug(
                    f"调用get_sequence_ids_by_sources，参数: {selected_sources}"
                )
                return self.storager.get_sequence_ids_by_sources(selected_sources)

            logger.debug("开始执行获取序列ID操作")
            sequence_ids = self._safe_duckdb_operation(get_sequence_ids)

            if sequence_ids is None:
                logger.warning("获取序列ID返回None，使用空列表")
                sequence_ids = []

            self.current_sequence_ids = [
                str(sid) for sid in sequence_ids if sid is not None
            ]

            logger.info(f"序列加载完成，找到 {len(sequence_ids)} 个序列")
            logger.debug(
                f"序列ID列表: {self.current_sequence_ids[:5]}{'...' if len(self.current_sequence_ids) > 5 else ''}"
            )

            if not sequence_ids:
                logger.warning("在选中的数据源中未找到任何序列")
                return "在选中的数据源中未找到任何序列", 0, 0

            return f"成功加载 {len(sequence_ids)} 个序列", 0, len(sequence_ids) - 1

        except Exception as e:
            logger.error(f"加载序列时出错: {str(e)}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            return f"加载序列时出错: {str(e)}", 0, 0

    def _get_sequence_info(self, seq_id: ID) -> dict:
        assert self.storager is not None

        def query_sequence_info():
            if self.storager is None:
                return []
            return self.storager.meta_handler.query_batch(
                "SELECT id, uri, source, meta FROM sequences WHERE id = ?",
                [seq_id.to_uuid()],
            )

        result = self._safe_duckdb_operation(query_sequence_info)
        return result[0] if result else {}

    def _sort_by_index(self, items: List[dict]) -> List[dict]:
        """按照index字段对项目进行排序，支持index_0, index_1等格式"""

        def extract_sort_key(item):
            index = item.get("index", "")
            if not index:
                return (1, float("inf"), "")  # 没有index的项目排在最后

            # 尝试提取数字部分
            import re

            match = re.search(r"(\d+)$", str(index))
            if match:
                # 数字结尾的index，按数字排序
                return (0, int(match.group(1)), str(index))
            else:
                # 非数字结尾的index，按字符串排序
                return (0, float("inf"), str(index))

        return sorted(items, key=extract_sort_key)

    def get_random_sequence(
        self, slider_value: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[str, str]], Any]:
        """获取随机序列的文本和图片数据"""
        if not self.storager or not self.current_sequence_ids:
            return pd.DataFrame(), pd.DataFrame(), [], "请先加载序列数据"

        try:
            # 根据滑块值选择序列
            if slider_value >= len(self.current_sequence_ids):
                slider_value = len(self.current_sequence_ids) - 1

            selected_sequence_id = self.current_sequence_ids[slider_value]
            sequence_id_obj = ID.from_uuid(selected_sequence_id)

            # 获取序列元数据
            seq_metas = self.storager.get_sequence_metas([sequence_id_obj])
            if not seq_metas:
                return (
                    pd.DataFrame(),
                    pd.DataFrame(),
                    [],
                    f"未找到序列 {selected_sequence_id} 的数据",
                )
            seq_info = self._get_sequence_info(sequence_id_obj)

            seq_meta = seq_metas[0]

            sequence_items = [
                dict(
                    id=seq_meta["sequence_id"],
                    type="sequence",
                    source=seq_info.get("source", ""),
                    index=None,
                    uri=seq_info.get("uri", ""),
                )
            ]

            # 处理文本数据
            texts_df = pd.DataFrame()
            if seq_meta.get("texts"):
                texts_data = []
                # 先对文本数据按index排序
                sorted_texts = self._sort_by_index(seq_meta["texts"])

                for text in sorted_texts:
                    texts_data.append(
                        {
                            "ID": str(text["id"]),
                            "索引": text.get("index", ""),
                            "内容": text["content"],
                        }
                    )
                    sequence_items.append(
                        dict(
                            id=text["id"],
                            type="text",
                            source=text.get("source", ""),
                            index=text.get("index", ""),
                            uri=text.get("uri", ""),
                        )
                    )
                texts_df = pd.DataFrame(texts_data)

            # 处理图片数据
            image_gallery = []
            if seq_meta.get("images"):
                # 先对图片数据按index排序
                sorted_images = self._sort_by_index(seq_meta["images"])

                # 获取图片数据
                image_ids = [img["id"] for img in sorted_images]
                image_bytes = self.storager.get_image_bytes_by_ids(image_ids)

                for img in sorted_images:
                    sequence_items.append(
                        dict(
                            id=img["id"],
                            type="image",
                            source=img.get("source", ""),
                            index=img.get("index", ""),
                            uri=img.get("uri", ""),
                        )
                    )

                    if img["id"] not in image_bytes:
                        pil_image = self.generate_notice_image()
                        caption = f"not found {img['id']}"
                        image_gallery.append((pil_image, caption))
                        continue

                    image_data = image_bytes[img["id"]]
                    pil_image = Image.open(io.BytesIO(image_data))

                    # 创建描述文本
                    caption = (
                        f"{img['id']}\t{img.get('width', -1)}x{img.get('height', -1)}"
                    )
                    image_gallery.append((pil_image, caption))

            sequence_df = pd.DataFrame(sequence_items)

            # 保存当前序列的所有元素（除了sequence本身），并按index排序
            self.current_sequence_items = self._sort_by_index(
                [item for item in sequence_items if item["type"] in ["text", "image"]]
            )

            # 保存当前文本数据用于HTML生成
            self.current_texts_df = texts_df

            return sequence_df, texts_df, image_gallery, seq_info.get("meta", {})

        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)
            return pd.DataFrame(), pd.DataFrame(), [], f"获取序列数据时出错: {str(e)}"

    def get_available_elements(self) -> List[Tuple[str, str]]:
        """获取当前序列中可选择的元素列表"""
        if not self.current_sequence_items:
            return []

        elements = []
        for item in self.current_sequence_items:
            element_id = item["id"]
            element_type = item["type"]
            uri = item.get("uri", "")
            # 确保element_id是字符串格式
            element_id_str = str(element_id)
            display_name = f"{element_type}: {element_id_str}"
            if uri:
                display_name += f" ({uri})"
            elements.append((display_name, element_id_str))

        return elements

    def add_selected_element(self, element_id: str) -> str:
        """添加选中的元素到排序列表"""
        if not self.current_sequence_items:
            return "没有可选择的元素"

        # 查找元素 - 使用字符串比较
        element = None
        for item in self.current_sequence_items:
            if str(item["id"]) == element_id:
                element = item
                break

        if not element:
            return f"未找到元素 {element_id}"

        # 检查是否已经添加过
        for selected in self.selected_elements:
            if str(selected["id"]) == element_id:
                return f"元素 {element_id} 已经添加过了"

        # 添加到选中列表
        self.selected_elements.append(element)
        return f"已添加元素: {element['type']} - {element_id}"

    def remove_selected_element(self, element_id: str) -> str:
        """从排序列表中移除元素"""
        for i, element in enumerate(self.selected_elements):
            if str(element["id"]) == element_id:
                removed_element = self.selected_elements.pop(i)
                return f"已移除元素: {removed_element['type']} - {element_id}"

        return f"未找到元素 {element_id}"

    def clear_selected_elements(self) -> str:
        """清空选中的元素列表"""
        count = len(self.selected_elements)
        self.selected_elements.clear()
        return f"已清空 {count} 个选中的元素"

    def generate_html_sequence(self) -> str:
        """根据选中的元素生成HTML序列"""
        if not self.selected_elements:
            return "<p>请先选择要展示的元素</p>"

        if not self.storager:
            return "<p>错误: 存储系统未初始化</p>"

        try:
            html_parts = []
            html_parts.append("""
            <div style="
                font-family: system-ui, -apple-system, sans-serif;
                line-height: 1.5;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
                background: #fafafa;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            ">
            """)

            for i, element in enumerate(self.selected_elements, 1):
                element_type = element["type"]
                element_id = str(element["id"])  # 确保ID是字符串格式
                element_uri = element.get("uri", "")
                element_source = element.get("source", "")
                element_index = element.get("index", "")

                if element_type == "text":
                    # 获取文本内容
                    text_content = self._get_text_content(element_id)
                    html_parts.append(f"""
                    <div style="
                        margin: 20px 0;
                        padding: 16px;
                        background: #f8f9fa;
                        border-radius: 6px;
                        border-left: 4px solid #28a745;
                    ">
                        <div style="
                            font-size: 16px;
                            color: #333;
                            white-space: pre-wrap;
                            word-wrap: break-word;
                            margin-bottom: 8px;
                        ">{text_content}</div>
                        <details style="
                            font-size: 11px;
                            color: #999;
                            margin-top: 8px;
                            padding-top: 6px;
                            border-top: 1px solid #f0f0f0;
                        ">
                            <summary style="cursor: pointer; color: #666;">📋 详细信息</summary>
                            <div style="margin-top: 4px; padding-left: 8px;">
                                ID: {element_id}<br>
                                来源: {element_source}<br>
                                索引: {element_index}
                                {f"<br>URI: {element_uri}" if element_uri else ""}
                            </div>
                        </details>
                    </div>
                    """)

                elif element_type == "image":
                    # 获取图片内容
                    image_html = self._get_image_html(element_id)
                    html_parts.append(f"""
                    <div style="
                        margin: 20px 0;
                        padding: 16px;
                        background: #fff3cd;
                        border-radius: 6px;
                        border-left: 4px solid #ffc107;
                    ">
                        <div style="
                            text-align: center;
                            margin: 12px 0;
                        ">{image_html}</div>
                        <details style="
                            font-size: 11px;
                            color: #999;
                            margin-top: 8px;
                            padding-top: 6px;
                            border-top: 1px solid #f0f0f0;
                        ">
                            <summary style="cursor: pointer; color: #666;">📋 详细信息</summary>
                            <div style="margin-top: 4px; padding-left: 8px;">
                                ID: {element_id}<br>
                                来源: {element_source}<br>
                                索引: {element_index}
                                {f"<br>URI: {element_uri}" if element_uri else ""}
                            </div>
                        </details>
                    </div>
                    """)

            html_parts.append("""
            </div>
            """)

            return "".join(html_parts)

        except Exception as e:
            return f"<p>生成HTML时出错: {str(e)}</p>"

    def _get_text_content(self, text_id: str) -> str:
        """获取文本内容"""
        try:
            # 从当前序列的文本数据中查找
            if hasattr(self, "current_texts_df") and not self.current_texts_df.empty:
                for _, row in self.current_texts_df.iterrows():
                    if str(row["ID"]) == text_id:
                        return str(row["内容"])
            return f"文本内容未找到 (ID: {text_id})"
        except Exception as e:
            return f"获取文本内容时出错: {str(e)}"

    def _get_image_html(self, image_id: str) -> str:
        """获取图片的HTML"""
        try:
            if not self.storager:
                return f"存储系统未初始化 (ID: {image_id})"

            # 从Lance表中获取图片数据
            from vault.schema import ID

            image_id_obj = ID.from_uuid(image_id)

            # Lance操作通常不需要DuckDB锁，但为了安全起见，我们也保护一下
            def get_image_data():
                if self.storager is None:
                    return None
                return self.lance_taker.by_ids(
                    self.storager.lance_uris["images"],
                    [image_id_obj],
                    columns=["id", "image", "width", "height"],
                )

            image_table = self._safe_duckdb_operation(get_image_data)

            if image_table is None:
                return f"存储系统未初始化 (ID: {image_id})"

            if image_table.num_rows > 0:
                row = image_table.take([0])
                image_data = row.column("image")[0].as_py()

                # 将图片数据转换为base64
                import base64

                image_base64 = base64.b64encode(image_data).decode("utf-8")

                return f'<img src="data:image/jpeg;base64,{image_base64}" alt="图片 {image_id}" style="max-width: 100%; height: auto;" />'
            else:
                return f"图片未找到 (ID: {image_id})"
        except Exception as e:
            return f"获取图片时出错: {str(e)}"


def set_defaults_from_url(request: gr.Request):
    """
    当页面加载时，从 URL 参数读取 vault_path, sources, sequence_index 的值，
    并用它们来更新相应的组件。
    """
    logger.info("页面加载，开始解析URL参数")
    logger.debug(f"请求URL: {request.url}")
    logger.debug(f"请求头: {dict(request.headers)}")

    vault_path = request.query_params.get("vault_path", "")
    sources_str = request.query_params.get("sources", "")
    sequence_index = request.query_params.get("sequence_index", "0")

    logger.info("URL参数解析结果:")
    logger.info(f"  - vault_path: {vault_path}")
    logger.info(f"  - sources: {sources_str}")
    logger.info(f"  - sequence_index: {sequence_index}")

    # 处理sources参数（可能是逗号分隔的字符串）
    sources = []
    if sources_str:
        sources = [s.strip() for s in sources_str.split(",") if s.strip()]
        logger.debug(f"解析后的sources列表: {sources}")

    try:
        sequence_index = int(sequence_index)
        logger.debug(f"序列索引解析成功: {sequence_index}")
    except (ValueError, TypeError) as e:
        logger.warning(f"序列索引解析失败，使用默认值0: {e}")
        sequence_index = 0

    logger.info("URL参数处理完成，准备更新组件状态")
    return (
        gr.update(value=vault_path),
        gr.update(choices=sources, value=sources, interactive=len(sources) > 0),
        gr.update(value=sequence_index),
    )


def generate_share_link(
    vault_path, selected_sources, sequence_index, request: gr.Request
):
    """
    根据当前组件的状态生成一个可分享的 URL 链接。
    """
    logger.info("开始生成分享链接")
    logger.debug("当前状态:")
    logger.debug(f"  - vault_path: {vault_path}")
    logger.debug(f"  - selected_sources: {selected_sources}")
    logger.debug(f"  - sequence_index: {sequence_index}")

    params = {}

    if vault_path:
        params["vault_path"] = vault_path
        logger.debug(f"添加vault_path参数: {vault_path}")

    if selected_sources:
        params["sources"] = ",".join(selected_sources)
        logger.debug(f"添加sources参数: {selected_sources}")

    if sequence_index is not None:
        params["sequence_index"] = int(sequence_index)
        logger.debug(f"添加sequence_index参数: {sequence_index}")

    if not params:
        logger.warning("没有有效参数，无法生成分享链接")
        return gr.HTML("<p>请先设置一些参数再生成分享链接</p>")

    # 获取基础URL
    referer_url = request.headers.get("referer")
    if referer_url:
        base_url = referer_url.split("?")[0]
        logger.debug(f"使用referer URL作为基础URL: {base_url}")
    else:
        base_url = f"{request.url.scheme}://{request.url.netloc}/"
        logger.debug(f"使用请求URL构建基础URL: {base_url}")

    query_string = urlencode(params)
    full_url = f"{base_url}?{query_string}"

    logger.info(f"分享链接生成成功: {full_url}")
    logger.debug(f"查询参数: {params}")

    html_link = f"""
    <div class="share-link-container">
        <p>🎉 分享链接已生成！点击下方链接分享：</p>
        <a href="{full_url}" target="_blank" class="share-link">{full_url}</a>
        <p style="margin-top: 10px; font-size: 0.9em; color: #666;">
            复制此链接分享给他人，他们打开后将自动加载相同的配置
        </p>
    </div>
    """
    return gr.HTML(html_link)


def create_interface():
    """创建Gradio界面"""
    browser = VaultBrowser()

    # 默认vault路径
    default_vault_path = (
        "/mnt/jfs/datasets/vault/composed_image_retrieval/20250905-cirr"
    )

    # 自定义CSS样式，让分享链接更醒目
    CUSTOM_CSS = """
    .share-link-container {
        padding: 15px;
        border-radius: 10px;
        background-color: #E8F5E9; /* 浅绿色背景 */
        border: 2px solid #4CAF50; /* 绿色边框 */
        text-align: center;
        margin-top: 20px;
    }
    .share-link-container p {
        margin: 0 0 10px 0;
        font-size: 1em;
        color: #2E7D32; /* 深绿色文字 */
    }
    .share-link {
        display: inline-block;
        padding: 10px 15px;
        background-color: #4CAF50; /* 绿色按钮背景 */
        color: white !important; /* 白色文字，!important确保覆盖默认样式 */
        text-decoration: none;
        border-radius: 5px;
        font-weight: bold;
        transition: background-color 0.3s ease;
        word-break: break-all;
    }
    .share-link:hover {
        background-color: #45a049; /* 悬停时颜色变深 */
    }
    """

    with gr.Blocks(
        title="Vault数据浏览器",
        theme=gradio.themes.Soft(),
        css=CUSTOM_CSS,
    ) as demo:
        gr.Markdown("# 🗄️ Vault数据浏览器")
        gr.Markdown("浏览和探索vault中的多模态数据")

        with gr.Accordion("操作说明", open=False):
            gr.Markdown("""
                1. 输入vault路径并点击"加载Vault"
                2. 选择要浏览的数据源
                3. 点击"加载序列"获取所有序列
                4. 使用滑块或"随机选择"浏览序列
                5. 查看下方的文本和图片内容
                """)

        with gr.Row():
            with gr.Column(scale=3):
                vault_path_input = gr.Textbox(
                    label="Vault路径",
                    value=default_vault_path,
                    placeholder="请输入vault数据路径...",
                    info="输入vault数据的根目录路径",
                )
            with gr.Column(scale=1):
                vault_status = gr.Textbox(
                    label="状态", value="等待加载vault...", interactive=False
                )
        load_vault_btn = gr.Button("加载Vault", variant="primary")

        with gr.Row():
            with gr.Column(scale=3):
                sources_dropdown = gr.Dropdown(
                    label="选择数据源",
                    choices=[],
                    multiselect=True,
                    info="选择一个或多个数据源",
                    interactive=False,
                )
            with gr.Column(scale=1):
                sequences_status = gr.Textbox(
                    label="序列状态", value="等待加载序列...", interactive=False
                )
        load_sequences_btn = gr.Button("加载序列", variant="primary")

        with gr.Row():
            with gr.Column(scale=3):
                sequence_slider = gr.Slider(
                    minimum=0,
                    maximum=0,
                    step=1,
                    value=0,
                    label="选择序列",
                    info="拖动滑块选择要查看的序列",
                    interactive=False,
                )
            with gr.Column(scale=1):
                random_btn = gr.Button("随机选择", variant="secondary")

        # 分享链接功能
        with gr.Row():
            with gr.Column(scale=3):
                share_btn = gr.Button("🔗 生成分享链接", variant="secondary")
            with gr.Column(scale=1):
                gr.Markdown("**分享当前配置**")

        share_output = gr.HTML()

        sequence_dataframe = gr.Dataframe(
            label="序列",
            headers=["type", "id", "source", "index", "uri"],
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
        )

        sequence_info = gr.JSON(label="序列meta")

        texts_dataframe = gr.Dataframe(
            label="文本数据",
            headers=["ID", "内容", "语言", "索引", "URI", "来源"],
            datatype=["str", "str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
        )

        images_gallery = gr.Gallery(
            label="图片数据",
            show_label=True,
            elem_id="gallery",
            columns=3,
            rows=4,
            object_fit="contain",
            allow_preview=True,
            show_share_button=True,
        )

        # 自定义序列生成部分
        gr.Markdown("---")
        gr.Markdown("## 🎨 自定义序列生成")
        gr.Markdown("选择并排序文本和图片元素，生成美观的HTML展示页面")

        with gr.Row():
            with gr.Column(scale=3):
                element_dropdown = gr.Dropdown(
                    label="选择元素",
                    choices=[],
                    info="从当前序列中选择要添加的元素",
                    interactive=False,
                )
            with gr.Column(scale=1):
                add_element_btn = gr.Button(
                    "添加元素", variant="secondary", interactive=False
                )

        with gr.Row():
            with gr.Column(scale=3):
                selected_elements_display = gr.Textbox(
                    label="已选择的元素",
                    value="",
                    interactive=False,
                    lines=3,
                    info="显示已选择的元素列表",
                )
            with gr.Column(scale=1):
                clear_elements_btn = gr.Button(
                    "清空列表", variant="stop", interactive=False
                )

        with gr.Row():
            generate_html_btn = gr.Button(
                "生成HTML序列", variant="primary", interactive=False
            )

        html_output = gr.HTML(
            label="生成的HTML序列",
            value="<p>请先选择元素，然后点击生成HTML序列</p>",
        )

        # 事件处理
        def on_load_vault(vault_path):
            logger.info(f"用户点击加载Vault按钮，路径: {vault_path}")
            status, sources = browser.load_vault(vault_path)
            logger.info(f"Vault加载完成，状态: {status}, 数据源数量: {len(sources)}")
            return (
                status,
                gr.Dropdown(choices=sources, interactive=len(sources) > 0),
                gr.Button(interactive=len(sources) > 0),
            )

        def on_load_vault_with_sources(vault_path, selected_sources):
            """加载vault并自动选择指定的sources"""
            status, sources = browser.load_vault(vault_path)

            # 如果URL中指定了sources，自动选择它们
            if selected_sources and sources:
                # 过滤出存在的sources
                valid_sources = [s for s in selected_sources if s in sources]
                if valid_sources:
                    # 自动加载序列
                    seq_status, min_val, max_val = browser.load_sequences(valid_sources)
                    if max_val <= min_val:
                        max_val = min_val + 1
                    return (
                        status,
                        gr.Dropdown(
                            choices=sources,
                            value=valid_sources,
                            interactive=len(sources) > 0,
                        ),
                        gr.Button(interactive=len(sources) > 0),
                        seq_status,
                        min_val,
                        max_val,
                        gr.Slider(
                            minimum=min_val, maximum=max_val, interactive=max_val > 0
                        ),
                    )

            return (
                status,
                gr.Dropdown(
                    choices=sources,
                    value=selected_sources if selected_sources else [],
                    interactive=len(sources) > 0,
                ),
                gr.Button(interactive=len(sources) > 0),
                "请选择数据源后点击加载序列",
                0,
                0,
                gr.Slider(interactive=False),
            )

        def on_load_sequences(selected_sources):
            logger.info(f"用户点击加载序列按钮，选中的数据源: {selected_sources}")

            if not selected_sources:
                logger.warning("未选择数据源")
                return "请先选择数据源", 0, 0, gr.Slider(interactive=False)

            status, min_val, max_val = browser.load_sequences(selected_sources)
            logger.info(f"序列加载完成，状态: {status}, 范围: {min_val}-{max_val}")

            if max_val <= min_val:
                logger.warning(
                    f"max_val <= min_val, max_val: {max_val}, min_val: {min_val}"
                )
                max_val = min_val + 1

            return (
                status,
                min_val,
                max_val,
                gr.Slider(minimum=min_val, maximum=max_val, interactive=max_val > 0),
            )

        def on_slider_change(slider_value):
            logger.info(f"用户拖动滑块，选择序列索引: {slider_value}")

            sequence_df, texts_df, image_gallery, info_text = (
                browser.get_random_sequence(slider_value)
            )
            logger.info(
                f"序列数据获取完成，文本数量: {len(texts_df)}, 图片数量: {len(image_gallery)}"
            )

            # 更新元素选择下拉框
            available_elements = browser.get_available_elements()
            elements_enabled = len(available_elements) > 0
            logger.debug(
                f"可用元素数量: {len(available_elements)}, 元素选择功能启用: {elements_enabled}"
            )

            return (
                sequence_df,
                texts_df,
                image_gallery,
                info_text,
                gr.Dropdown(choices=available_elements, interactive=elements_enabled),
                gr.Button(interactive=elements_enabled),
                gr.Button(interactive=len(browser.selected_elements) > 0),
                gr.Button(interactive=len(browser.selected_elements) > 0),
                _format_selected_elements_display(),
            )

        def on_random_select():
            logger.info("用户点击随机选择按钮")

            if browser.current_sequence_ids:
                random_index = random.randint(0, len(browser.current_sequence_ids) - 1)
                logger.info(
                    f"随机选择序列索引: {random_index} (总序列数: {len(browser.current_sequence_ids)})"
                )

                sequence_df, texts_df, image_gallery, info_text = (
                    browser.get_random_sequence(random_index)
                )
                logger.info(
                    f"随机序列数据获取完成，文本数量: {len(texts_df)}, 图片数量: {len(image_gallery)}"
                )

                # 更新元素选择下拉框
                available_elements = browser.get_available_elements()
                elements_enabled = len(available_elements) > 0
                logger.debug(f"可用元素数量: {len(available_elements)}")

                return (
                    random_index,
                    sequence_df,
                    texts_df,
                    image_gallery,
                    info_text,
                    gr.Dropdown(
                        choices=available_elements, interactive=elements_enabled
                    ),
                    gr.Button(interactive=elements_enabled),
                    gr.Button(interactive=len(browser.selected_elements) > 0),
                    gr.Button(interactive=len(browser.selected_elements) > 0),
                    _format_selected_elements_display(),
                )

            logger.warning("没有可用的序列ID，无法进行随机选择")
            return (
                0,
                pd.DataFrame(),
                pd.DataFrame(),
                [],
                "请先加载序列数据",
                gr.Dropdown(choices=[], interactive=False),
                gr.Button(interactive=False),
                gr.Button(interactive=False),
                gr.Button(interactive=False),
                "请先加载序列数据",
            )

        def _format_selected_elements_display():
            """格式化已选择元素的显示文本"""
            if not browser.selected_elements:
                return "暂无选择的元素"

            display_lines = []
            for i, element in enumerate(browser.selected_elements, 1):
                element_type = element["type"]
                element_id = str(element["id"])  # 确保ID是字符串格式
                element_uri = element.get("uri", "")
                display_text = f"{i}. {element_type}: {element_id}"
                if element_uri:
                    display_text += f" ({element_uri})"
                display_lines.append(display_text)

            return "\n".join(display_lines)

        def on_add_element(selected_element_id):
            """添加选中的元素"""
            if not selected_element_id:
                return (
                    "请先选择一个元素",
                    _format_selected_elements_display(),
                    gr.Button(interactive=len(browser.selected_elements) > 0),
                    gr.Button(interactive=len(browser.selected_elements) > 0),
                )

            result = browser.add_selected_element(selected_element_id)
            return (
                result,
                _format_selected_elements_display(),
                gr.Button(interactive=len(browser.selected_elements) > 0),  # 清空按钮
                gr.Button(
                    interactive=len(browser.selected_elements) > 0
                ),  # 生成HTML按钮
            )

        def on_clear_elements():
            """清空选择的元素"""
            result = browser.clear_selected_elements()
            return (
                result,
                _format_selected_elements_display(),
                gr.Button(interactive=len(browser.selected_elements) > 0),  # 清空按钮
                gr.Button(
                    interactive=len(browser.selected_elements) > 0
                ),  # 生成HTML按钮
            )

        def on_generate_html():
            """生成HTML序列"""
            html_content = browser.generate_html_sequence()
            return html_content

        def on_share_link_click(
            vault_path, selected_sources, sequence_index, request: gr.Request
        ):
            """处理分享链接按钮点击"""
            logger.info("用户点击生成分享链接按钮")
            logger.debug(
                f"当前状态: vault_path={vault_path}, sources={selected_sources}, index={sequence_index}"
            )

            result = generate_share_link(
                vault_path, selected_sources, sequence_index, request
            )

            logger.info("分享链接生成完成")
            return result

        def on_url_load(request: gr.Request):
            """从URL参数加载初始状态"""
            logger.info("开始从URL参数加载初始状态")
            logger.debug(f"请求URL: {request.url}")

            vault_path = request.query_params.get("vault_path", "")
            sources_str = request.query_params.get("sources", "")
            sequence_index = request.query_params.get("sequence_index", "0")

            logger.info("URL参数解析:")
            logger.info(f"  - vault_path: {vault_path}")
            logger.info(f"  - sources: {sources_str}")
            logger.info(f"  - sequence_index: {sequence_index}")

            # 处理sources参数
            sources = []
            if sources_str:
                sources = [s.strip() for s in sources_str.split(",") if s.strip()]
                logger.debug(f"解析后的sources: {sources}")

            try:
                sequence_index = int(sequence_index)
                logger.debug(f"序列索引: {sequence_index}")
            except (ValueError, TypeError) as e:
                logger.warning(f"序列索引解析失败: {e}")
                sequence_index = 0

            # 如果没有vault_path，直接返回默认状态
            if not vault_path:
                logger.info("没有vault_path参数，返回默认状态")
                return (
                    gr.update(value=""),
                    gr.update(choices=[], value=[], interactive=False),
                    gr.update(value=0),
                    "等待加载vault...",
                    gr.Dropdown(choices=[], interactive=False),
                    gr.Button(interactive=False),
                    "等待加载序列...",
                    0,
                    0,
                    gr.Slider(interactive=False),
                )

            # 加载vault和sources
            logger.info(f"开始加载vault: {vault_path}")
            status, available_sources = browser.load_vault(vault_path)
            logger.info(f"Vault加载结果: {status}")

            if not available_sources:
                logger.warning("未找到可用的数据源")
                return (
                    gr.update(value=vault_path),
                    gr.update(choices=[], value=[], interactive=False),
                    gr.update(value=sequence_index),
                    status,
                    gr.Dropdown(choices=[], interactive=False),
                    gr.Button(interactive=False),
                    "等待加载序列...",
                    0,
                    0,
                    gr.Slider(interactive=False),
                )

            # 如果有指定的sources，自动加载序列
            if sources:
                valid_sources = [s for s in sources if s in available_sources]
                logger.info(f"验证sources: {sources} -> {valid_sources}")

                if valid_sources:
                    logger.info(f"自动加载序列，sources: {valid_sources}")
                    seq_status, min_val, max_val = browser.load_sequences(valid_sources)
                    logger.info(
                        f"序列加载结果: {seq_status}, 范围: {min_val}-{max_val}"
                    )

                    if max_val <= min_val:
                        max_val = min_val + 1
                        logger.warning(f"调整max_val: {max_val}")

                    # 确保sequence_index在有效范围内
                    if sequence_index > max_val:
                        logger.warning(
                            f"序列索引超出范围，调整: {sequence_index} -> {max_val}"
                        )
                        sequence_index = max_val
                    elif sequence_index < min_val:
                        logger.warning(
                            f"序列索引低于范围，调整: {sequence_index} -> {min_val}"
                        )
                        sequence_index = min_val

                    logger.info(
                        f"URL加载完成，最终状态: vault={vault_path}, sources={valid_sources}, index={sequence_index}"
                    )
                    return (
                        gr.update(value=vault_path),
                        gr.update(
                            choices=available_sources,
                            value=valid_sources,
                            interactive=True,
                        ),
                        gr.update(value=sequence_index),
                        status,
                        gr.Dropdown(
                            choices=available_sources,
                            value=valid_sources,
                            interactive=True,
                        ),
                        gr.Button(interactive=True),
                        seq_status,
                        min_val,
                        max_val,
                        gr.Slider(
                            minimum=min_val,
                            maximum=max_val,
                            value=sequence_index,
                            interactive=True,
                        ),
                    )

            logger.info(f"URL加载完成，部分状态: vault={vault_path}, sources={sources}")
            return (
                gr.update(value=vault_path),
                gr.update(choices=available_sources, value=sources, interactive=True),
                gr.update(value=sequence_index),
                status,
                gr.Dropdown(choices=available_sources, value=sources, interactive=True),
                gr.Button(interactive=True),
                "请选择数据源后点击加载序列",
                0,
                0,
                gr.Slider(interactive=False),
            )

        # 绑定事件
        load_vault_btn.click(
            fn=on_load_vault,
            inputs=[vault_path_input],
            outputs=[vault_status, sources_dropdown, load_sequences_btn],
        )

        load_sequences_btn.click(
            fn=on_load_sequences,
            inputs=[sources_dropdown],
            outputs=[
                sequences_status,
                gr.Number(visible=False),
                gr.Number(visible=False),
                sequence_slider,
            ],
        )

        sequence_slider.change(
            fn=on_slider_change,
            inputs=[sequence_slider],
            outputs=[
                sequence_dataframe,
                texts_dataframe,
                images_gallery,
                sequence_info,
                element_dropdown,
                add_element_btn,
                clear_elements_btn,
                generate_html_btn,
                selected_elements_display,
            ],
        )

        random_btn.click(
            fn=on_random_select,
            inputs=[],
            outputs=[
                sequence_slider,
                sequence_dataframe,
                texts_dataframe,
                images_gallery,
                sequence_info,
                element_dropdown,
                add_element_btn,
                clear_elements_btn,
                generate_html_btn,
                selected_elements_display,
            ],
        )

        # 新的事件绑定
        add_element_btn.click(
            fn=on_add_element,
            inputs=[element_dropdown],
            outputs=[
                gr.Textbox(visible=False),
                selected_elements_display,
                clear_elements_btn,
                generate_html_btn,
            ],
        )

        clear_elements_btn.click(
            fn=on_clear_elements,
            inputs=[],
            outputs=[
                gr.Textbox(visible=False),
                selected_elements_display,
                clear_elements_btn,
                generate_html_btn,
            ],
        )

        generate_html_btn.click(
            fn=on_generate_html,
            inputs=[],
            outputs=[html_output],
        )

        # 分享链接按钮事件
        share_btn.click(
            fn=on_share_link_click,
            inputs=[vault_path_input, sources_dropdown, sequence_slider],
            outputs=[share_output],
        )

        # 页面加载时从URL参数初始化
        demo.load(
            fn=on_url_load,
            inputs=None,
            outputs=[
                vault_path_input,
                sources_dropdown,
                sequence_slider,
                vault_status,
                sources_dropdown,  # 这里会更新choices
                load_sequences_btn,
                sequences_status,
                gr.Number(visible=False),  # min_val
                gr.Number(visible=False),  # max_val
                sequence_slider,  # 这里会更新slider的range
            ],
        )

    return demo


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("启动Vault数据浏览器")
    logger.info("=" * 50)

    demo = create_interface()
    logger.info("Gradio界面创建完成")

    logger.info("启动服务器...")
    logger.info("服务器地址: http://0.0.0.0:8078")
    logger.info("日志级别: DEBUG (详细调试信息)")
    logger.info("=" * 50)

    demo.launch(server_name="0.0.0.0", server_port=8078, share=False, show_error=True)
