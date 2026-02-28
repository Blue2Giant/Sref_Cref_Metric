import importlib.resources as pkg_resources
import sys
from ctypes import CDLL, POINTER, byref, c_float, c_uint8, memmove, sizeof
from typing import Optional, Tuple

import numpy as np
import PIL.Image
from loguru import logger


class PDQHasher:
    # 定义一些常量以提高代码可读性
    IMAGE_WIDTH = 512
    IMAGE_HEIGHT = 512
    HASH_LENGTH_BYTES = 32  # PDQ 哈希是 256 位，即 32 字节

    def __init__(self):
        so_name = "libyume_pdq.so"

        package_name = __package__
        try:
            so_file_ref = pkg_resources.files(package_name).joinpath(so_name)
            with pkg_resources.as_file(so_file_ref) as lib_path:
                self._lib = CDLL(lib_path)

        except FileNotFoundError:
            logger.error(
                f"Error: {so_name} not found within the package.", file=sys.stderr
            )
            # 在这里处理错误，例如抛出异常
            raise ImportError("Could not find the required .so file.")

        # 配置 C 函数的签名（参数类型和返回类型）
        self._hash_smart_kernel = self._lib.yume_pdq_hash_smart_kernel
        self._hash_smart_kernel.restype = c_float
        self._hash_smart_kernel.argtypes = [
            POINTER(c_float),  # input
            POINTER(c_float),  # threshold
            POINTER(c_uint8),  # output
            POINTER(c_float),  # buf1
            POINTER(c_float),  # tmp
            POINTER(c_float),  # pdqf
        ]

        # 预先分配内存缓冲区，以便在多次调用中复用
        buffer_size = self.IMAGE_WIDTH * self.IMAGE_HEIGHT
        self._input_buffer = (c_float * buffer_size)()
        self._output_buffer = (c_uint8 * self.HASH_LENGTH_BYTES)()

        # 内部工作缓冲区
        self._threshold = c_float(0.0)
        self._buf1 = (c_float * (128 * 128))()
        self._tmp = (c_float * 128)()
        self._pdqf = (c_float * (16 * 16))()

    def __call__(self, pil_image: PIL.Image.Image) -> Tuple[float, Optional[bytes]]:
        """
        计算给定 PIL 图像的 PDQ 哈希和质量。

        Args:
            pil_image (PIL.Image.Image): 输入图像。必须是 512x512 尺寸的
                                         灰度图 ('L' 模式)。

        Returns:
            Tuple[float, Optional[str]]: 一个包含两个元素的元组：
                - quality (float): 图像的质量分数。
                - hash (str | None): 计算出的 256 位十六进制哈希字符串。
                                     如果质量分数低于 0.5, 则返回 None。
        """
        # --- 输入验证 ---
        if pil_image.size != (self.IMAGE_WIDTH, self.IMAGE_HEIGHT):
            raise ValueError(
                f"输入图像尺寸必须是 ({self.IMAGE_WIDTH}, {self.IMAGE_HEIGHT}), "
                f"但实际为 {pil_image.size}."
            )
        if pil_image.mode != "L":
            raise ValueError(
                f"输入图像模式必须是 'L' (灰度), 但实际为 '{pil_image.mode}'."
            )

        # --- 数据准备 ---
        # 将 PIL 图像转换为 float32 类型的 numpy 数组
        img_array = np.array(pil_image, dtype=np.float32)

        # 使用 memmove 将 numpy 数组的数据高效地复制到 C 类型的输入缓冲区
        memmove(self._input_buffer, img_array.ctypes.data, sizeof(self._input_buffer))

        # --- 调用 C 函数 ---
        quality = self._hash_smart_kernel(
            self._input_buffer,
            byref(self._threshold),
            self._output_buffer,
            self._buf1,
            self._tmp,
            self._pdqf,
        )

        # --- 结果处理 ---
        # 将 C 字节缓冲区转换为十六进制字符串
        hash_bytes = bytes(self._output_buffer)

        return quality, hash_bytes if quality > 0.5 else None


pdq_hasher = PDQHasher()
