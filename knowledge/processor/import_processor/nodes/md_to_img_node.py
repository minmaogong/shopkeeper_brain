from dataclasses import dataclass

from knowledge.processor.import_processor.base import BaseNode, setup_logging, T
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError

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

class _ImageScanner:
    """
    主要职责：
    1. 根据图片目录，得到该目录下有效的图片文件
    2. 从md文件中定位图片的位置
    3. 获取该图片在md中的上下文内容（给VLM模型提供上下文信息，帮助 模型识别结果更加准确）
    4. 最终组装所有图片的上下文内容（List）
    """

class _VLMSummarizer:
    """
    主要职责：
    主要根据每一张图片信息以及每一张图片的上下文信息，生成对应图片的摘要信息
    """

class _ImageUploader:
    """"
    主要职责：
    1. 将本地图片上传至Minio，得到该图片在Minio中可访问的远程地址
    2. 替换md中的摘要和图片地址
    """

class MarkDownToImgNode(BaseNode):
    """
    主要职责：
    1. 得到四个类的实例对象
    2. 分别调用四个实例对象的处理方法
    """

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        入口逻辑
        :param state:
        :return:
        """



