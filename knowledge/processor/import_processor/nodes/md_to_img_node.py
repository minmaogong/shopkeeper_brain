from dataclasses import dataclass
from logging import Logger
from typing import Tuple, List, Dict, Set
from pathlib import Path

from knowledge.processor.import_processor.base import BaseNode, setup_logging, T
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError, FileProcessingError

@dataclass          # 未来直接实例化（不需要重写__init__方法、 __repr__方法）
class ImageContext:
    """
    一张图片上下文信息

    """
    head: str # 上文标题内容
    pre_text: str # 上文内容
    post_text: str # 下文内容

@dataclass
class ImageInfo:
    """
    一张图片的完整信息
    图片名字：作为存储图片摘要的字典容器key
    图片地址：1.vlm模型识别图片的输入 2.minio 存储图片的地址
    图片上下文信息：给vlm模型提供上下文信息，帮助模型识别
    """
    name: str # 图片名称(全名)
    path: str # 图片地址
    image_context: ImageContext # 图片上下文信息



class _MdFileHandler:
    """
    主要职责：
    1. 读取md内容、md_path、图片目录
    2. 备份新的md_content(方便测试观察)
    """
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def validate_and_read_md(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        """
        核心逻辑：
        1. 读取md内容
        2. 读取md的路径
        3. 读取图片目录

        Args:
            state: 上一个节点更新后的state

        Returns:
            Tuple[str, Path, Path]

        """

        # 1. 从state中获取md_path
        md_path = state.get("md_path", "")

        # 2. 非空判断
        if not md_path:
            raise StateFieldError(node_name=self.node_name, field_name="md_path", expected_type=str)

        # 3. Path 标准化
        md_path_obj = Path(md_path)

        # 4. 判断md_path路径是否存在
        if not md_path_obj.exists():
            raise StateFieldError(node_name=self.node_name, field_name="md_path", expected_type=Path
                                , message=f"md文件路径不存在: {md_path}")

        # 5. 读取md_content
        try:
            with open(md_path_obj, "r", encoding="utf-8") as f:
                md_content = f.read()
        except IOError as e:
            self.logger.error(f"MD文件：{md_path_obj.name} 打开失败")
            raise FileProcessingError(node_name=self.node_name, message=f"md文件打开失败: {md_path}, 错误信息: {e}")

        # 6. 获取图片目录
        img_dir = md_path_obj.parent / "images"

        # 7. 返回
        return md_content, md_path_obj, img_dir

class _ImageScanner:
    """
    主要职责：
    1. 根据图片目录，得到该目录下有效的图片文件
    2. 从md文件中定位图片的位置
    3. 获取该图片在md中的上下文内容（给VLM模型提供上下文信息，帮助 模型识别结果更加准确）
    4. 最终组装所有图片的上下文内容（List）
    """
    def __init__(self, logger: Logger):
        self.logger = logger

class _VLMSummarizer:
    """
    主要职责：
    主要根据每一张图片信息以及每一张图片的上下文信息，生成对应图片的摘要信息
    """
    def __init__(self, logger: Logger):
        self.logger = logger

class _ImageUploader:
    """"
    主要职责：
    1. 将本地图片上传至Minio，得到该图片在Minio中可访问的远程地址
    2. 替换md中的摘要和图片地址
    """
    def __init__(self, logger: Logger):
        self.logger = logger

class MarkDownToImgNode(BaseNode):
    """
    主要职责：
    1. 得到四个类的实例对象
    2. 分别调用四个实例对象的处理方法
    """

    name = "md_to_img_node"

    def __init__(self):
        super().__init__() # 显示调用父类的构造方法
        self._md_file_handler = _MdFileHandler(self.logger)
        self._image_scanner = _ImageScanner(self.logger)
        self._vlm_summarizer = _VLMSummarizer(self.logger)
        self._image_uploader = _ImageUploader(self.logger)

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        入口逻辑
        :param state:
        :return:
        """

        # 1. 操作md_file_handler
        self.log_step("step1", "读取MD内容、路径以及图片的目录")
        md_content, md_path_obj, img_dir_obj = self._md_file_handler.validate_and_read_md(state)
        # 1.1 判断图片目录不存在
        if not img_dir_obj.exists():
            state["md_content"] = md_content
            return state

        return state


