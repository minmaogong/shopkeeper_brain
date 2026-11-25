import base64
import re
import time
from collections import deque
from dataclasses import dataclass
from logging import Logger
from typing import Tuple, List, Dict, Set, Optional, Deque
from pathlib import Path

from openai import OpenAI

from knowledge.processor.import_processor.base import BaseNode, setup_logging, T
from knowledge.processor.import_processor.state import ImportGraphState, create_default_state
from knowledge.processor.import_processor.exceptions import StateFieldError, FileProcessingError, ImageProcessingError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


@dataclass  # 未来直接实例化（不需要重写__init__方法、 __repr__方法）
class ImageContext:
    """
    一张图片上下文信息

    """
    head: str  # 上文标题内容
    pre_text: str  # 上文内容
    post_text: str  # 下文内容


@dataclass
class ImageInfo:
    """
    一张图片的完整信息
    图片名字：作为存储图片摘要的字典容器key
    图片地址：1.vlm模型识别图片的输入 2.minio 存储图片的地址
    图片上下文信息：给vlm模型提供上下文信息，帮助模型识别
    """
    name: str  # 图片名称(全名)
    path: str  # 图片地址
    image_context: ImageContext  # 图片上下文信息


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

    def backup(self, md_path_obj: Path, new_md_content: str) -> str:
        new_file_path = md_path_obj.with_name( # with_name() 方法用于创建一个新的 Path 对象，表示与原始 md_path_obj 相同目录下的一个新文件，文件名为原始文件名加上 "_new" 后缀
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )
        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至：{new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise ImageProcessingError(f"文件写入失败：{e}", node_name=self.node_name)

        return str(new_file_path)


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

    def scan_imgs_dir(self, img_dir_obj: Path, md_content: str, md_path_obj: Path, image_extensions: Set[str],
                      img_content_length: int) -> List[ImageInfo]:
        """
        核心逻辑：
        1. 扫描指定图片目录下的所有图片文件

        2. 遍历每一个图片文件去MD文件中获取到位置（上下文）
        2.1 上文信息（标题 + 上文内容）
        2.2 下文信息（下文内容）

        3. 将每一个（图片的上下文：ImageContext）放到最终封装每一个图片完整信息(ImageInfo)的容器中

        4. 将容器返回
        Args:
            img_dir_obj: 图片目录
            md_content: MD内容
            md_path_obj: MD文件路径
            image_extensions: 允许的图片扩展名集合
            img_content_length: 上下文的长度（各自最大不能超过img_content_length）

        Returns:
            List[ImageInfo]: 所有图片的完整信息

        """

        img_info_list = []

        # 1. 遍历图片目录
        for img_path in img_dir_obj.iterdir():

            # 1.1 过滤掉子目录
            if not img_path.is_file():
                self.logger.error(f"{img_path}不是一个有效的文件")
                continue

            # 1.2 过滤掉不合法的图片文件
            if img_path.suffix not in image_extensions:
                self.logger.error(f"{img_path.suffix}不是允许的图片后缀格式")
                continue

            # 1.3 找该图片的上下文
            ctx = self._find_context(img_path.name, md_content, img_content_length)

            if not ctx:
                self.logger.info(f"MD中未找到该图片{img_path.name}引用")
                continue

            # 1.4 封装ImageInfo对象，并且添加到容器中
            img_info_list.append(ImageInfo(
                name=img_path.name,
                path=str(img_path),
                image_context=ctx
            ))

        self.logger.info(f"MD中共找到{len(img_info_list)}张图片的上下文信息")

        # 2. 返回所有图片的完整信息
        return img_info_list

    def _find_context(self, img_name: str, md_content: str, img_content_length: int) -> Optional[
        ImageContext]:  # Optional[ImageContext] 表示返回值可以是ImageContext类型，也可以是None
        """
        核心逻辑：
        查找图片的上下文
        Args:
            img_name: 图片名
            md_content: MD内容
            img_content_length: 上下文的长度（各自最大不能超过img_content_length）

        Returns:
            找到了 -----> ImageContext: 图片上下文信息
            没找到 -----> None
        """

        # 1. 预编译正则规则(主要目的：从MD中抓取到当前图片)
        # ![](images/xxx.png "abc") # 如果存在abc，鼠标悬停在图片上会显示abc
        # 正则载大模型应用中特别多
        # 匹配特殊字符需要使用转移字符 \[ \] \( \)， 也可以使用re.escape()函数来转义特殊字符
        pattern = re.compile(r'!\[.*?\]\(.*?' + re.escape(
            img_name) + r'.*?\)')  # .表示匹配任意字符，*表示匹配前面的字符0次或多次，?表示非贪婪模式，re.escape()用于转义内容里面的特殊字符

        # 2. 按行切割md_content
        md_lines = md_content.split("\n")

        # 3. 遍历每一行以及对应的行索引
        for md_idx, md_line in enumerate(md_lines):
            # 3.1 当前行不是当前图片
            if not pattern.search(md_line):
                continue

            # 3.2 当前行是当前图片
            # 上文
            # 上文标题的索引作为起始索引，图片所在行的索引作为结束索引
            head, prev_index = self._find_heading_up(md_lines, md_idx)
            pre_lines = md_lines[prev_index + 1:md_idx]  # 左闭右开区间，取出上文内容
            pre_context = self._extract_limited_context(pre_lines, img_content_length, direction="front")

            # 下文
            # 图片所在的索引作为起始索引，下文标题的索引作为结束索引
            next_index = self._find_heading_down(md_lines, md_idx)
            next_lines = md_lines[md_idx + 1:next_index]  # 左闭右开区间，取出下文内容
            post_context = self._extract_limited_context(next_lines, img_content_length, direction="back")

            return ImageContext(
                head=head,
                pre_text=pre_context,
                post_text=post_context
            )

        return None

    def _find_heading_up(self, md_lines: List[str], from_idx: int) -> Tuple[str, int]:
        """

        Args:
            md_lines: 整个MD内容按行切割后的列表
            from_idx: 从该索引开始向上查找标题

        Returns:
            当前图片最近的上文标题内容+索引
        """
        for i in range(from_idx-1, -1, -1): # 从from_idx-1开始向上查找标题, 到索引0结束，step=-1表示向上查找
            if re.match(r'^#{1,6}\s+', md_lines[i]): # 匹配标题的正则表达式：^ 表示行的开头，#{1,6}表示匹配1到6个#，\s表示匹配空格，+表示匹配前面的字符1次或多次
                return md_lines[i], i

        return "", -1  # 没有找到标题，返回空字符串和-1

    def _find_heading_down(self, md_lines: List[str], from_idx: int) -> int:
        """

        Args:
            md_lines: 整个MD内容按行切割后的列表
            from_idx: 从该索引开始向下查找标题

        Returns:
            当前图片最近的下文标题索引
        """
        for i in range(from_idx+1, len(md_lines)):
            if re.match(r'^#{1,6}\s+', md_lines[i]):
                return i

        return len(md_lines)  # 没有找到标题，返回MD内容的长度，表示下文没有标题

    def _extract_limited_context(self, extracted_md_lines: List[str], img_content_length: int, direction: str) -> str:
        """
        主要职责：截取给定的上下文内容
        截取策略：不直接根据字符数暴力截取，采用段落方式截取。最后根据段落的字符数是否达到最大上下文长度 选择留取
        段落的规则：
        1. 自然而然的段落：获取切分后的内容 如果是“”空字符串
        2. 人为设计其他图片作为段落（其他图片不要）
        Args:
            extracted_md_lines: 上（下）文
            img_content_length: 上下文的长度（各自最大不能超过img_content_length）
            direction: 方向 (向上找)

        Returns:
            str: 上（下）文内容

        """
        current_paragraph = []
        paragraphs = []
        # 1. 遍历截取的行
        for line in extracted_md_lines:

            # 1.1 定义自然而然段落的规则
            is_blank_line = not line.strip()  # 空行
            # 1.2 定义人为设计的图片段落规则
            is_other_image = re.match(r'^!\[.*?\]\(.*?\)$', line.strip())  # 其他图片 正则： ^! 表示行开头是!, $ 表示行结尾
            # 1.3 当前行是空行或者其他图片行
            if is_blank_line or is_other_image:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []
                continue
            # 1.4 当前行不是空行或者其他图片行
            current_paragraph.append(line)

        # 2. 处理最后的段落
        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        # 反转(就近原则: 如果是向上找，应该从下往上找段落)
        if direction == "front":
            paragraphs.reverse()

        # 3. 遍历段落列表(判断长度，以及最终选择留下哪些段落)
        total = 0
        selected = [] # 最终留下的段落列表
        for paragraph in paragraphs:
            if total + len(paragraph) > img_content_length and selected:
                break

            selected.append(paragraph)
            total += len(paragraph)

        # 反转（保证收集到的顺序和原文档中顺序一致，方便VLM参考）
        if direction == "front":
            selected.reverse()

        # 4. 将最终段落列表中的段落转成一个字符串
        return "\n\n".join(selected)  # 段落之间用两个换行符分隔


class _VLMSummarizer:
    """
    主要职责：
    主要根据每一张图片信息以及每一张图片的上下文信息，生成对应图片的摘要信息
    """

    def __init__(self, logger: Logger, requests_per_minute: int):
        self.logger = logger
        self.requests_per_minute = requests_per_minute

    def summary_all(self, document_name: str, img_info_list: List[ImageInfo], vl_model: str) -> Dict[str, str]:
        """
        职责：为所有图片生成摘要
        Args:
            document_name: 文档的名字
            img_info_list: 所有图片信息
            vl_model: vlm 模型名字

        Returns:
            Dict[str, str]: {“img_name”: "summary"}
        """

        summaries = {}
        request_timestamps: Deque[float] = deque()

        # 1. 获取VLM客户端
        try:
            vlm_client = AIClients.get_openai()
        except Exception as e:
            self.logger.error(f"获取VLM客户端失败: {e}")
            # 兼容
            for img_info in img_info_list:
                summaries[img_info.name] = "暂无摘要"
            return summaries

        # 2. 调用VLM 为每一张图片生成摘要
        for img_info in img_info_list:
            self._enforce_rate_limit(request_timestamps, self.requests_per_minute, window=60)
            summaries[img_info.name] = self._summary_one(document_name, img_info, vlm_client, vl_model)

        self.logger.info(f"生成{len(summaries)}图片摘要")
        return summaries

    def _summary_one(self, document_name: str, img_info: ImageInfo, vlm_client: OpenAI, vl_model: str) -> str:
        """
        调用VLM模型为当前图片生成摘要信息
        Args:
            document_name: MD文档的名字
            img_info: 当前图片的完整信息
            vlm_client: VLM客户端
            vl_model: vlm 模型名字

        Returns:
            str: 当前图片的摘要信息
        """

        # 1. 构造VLM需要的上下文 （标题名、上文内容、下文内容）
        parts = [p for p in (img_info.image_context.head, img_info.image_context.pre_text, img_info.image_context.post_text) if p]

        # 2. 构建最终的上下文
        final_context = "\n".join(parts) if parts else "暂无上下文"

        # 3. 根据图片地址获取到图片的内容（二进制字节流） ---文本协议认识（base64编码） ----> 解码（utf-8） ----> 字符串（文本协议能传输）----- 根据收到字符串解码（二进制字节流 还原图片内容）
        try:
            with open(img_info.path, "rb") as f: # rb 表示以二进制读取文件
                img_data = base64.b64encode(f.read()).decode("utf-8")  # base64编码，返回bytes类型，再解码成字符串
        except IOError as e:
            self.logger.error(f"读取图片文件 {img_info.path} 内容失败: {e}")
            return "暂无图片描述"

        # 4. 利用vlm客户端调用VLM模型
        try:
            resp = vlm_client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_name}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题摘要（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data}" # data:image/jpeg;base64, 表示这是一个base64编码的jpeg图片
                            },
                        },
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()  # 去掉首尾空格
        except Exception as e:
            self.logger.error(f"图片摘要生成失败 {img_info.path}: {e}")
            return "暂无图片描述"

    def _enforce_rate_limit(self, timestamps: Deque[float], max_requests: int, window: int = 60):
        """
        强制执行速率限制，确保在给定的时间窗口内不会超过最大请求数。
        该方法会检查请求时间戳队列，如果在指定的时间窗口内请求次数超过了最大请求数，则会暂停执行，直到可以继续发送请求为止。
        Args:
            timestamps: 请求时间戳的双端队列
            max_requests: 最大请求数
            window: 时间窗口（秒）作用是限制在这个时间窗口内的请求次数不超过max_requests

        Returns:

        """
        now = time.time() # 获取当前时间戳（单位：秒）
        while timestamps and now - timestamps[0] >= window: # 如果队列不为空，并且当前时间戳减去队列中最早的时间戳大于等于窗口时间，则说明最早的请求已经超出了时间窗口，可以移除
            timestamps.popleft() # 移除队列中最早的时间戳

        if len(timestamps) >= max_requests: # 如果队列中请求次数已经达到最大请求数，则需要暂停执行，直到可以继续发送请求为止
            sleep_dur = window - (now - timestamps[0]) # 计算需要暂停的时间，等于窗口时间减去当前时间戳与队列中最早的时间戳的差值
            if sleep_dur > 0: # 如果需要暂停的时间大于0，则说明还需要等待一段时间才能继续发送请求
                self.logger.info(f"达到速率限制，暂停 {sleep_dur:.2f} 秒...")
                time.sleep(sleep_dur) # 暂停执行，等待一段时间

            now = time.time() # 更新当前时间戳
            while timestamps and now - timestamps[0] >= window: # 再次检查队列中最早的时间戳是否已经超出了时间窗口，如果是，则移除
                timestamps.popleft() # 移除队列中最早的时间戳

        timestamps.append(now) # 将当前时间戳添加到队列中，表示已经发送了一个请求

class _ImageUploader:
    """
    主要职责：
    1. 将本地图片上传至Minio，得到该图片在Minio中可访问的远程地址
    2. 替换md中的摘要和图片地址
    """

    def __init__(self, logger: Logger):
        self.logger = logger

    def upload_and_replace(self, object_dir_name: str, md_content: str, img_info_list: List[ImageInfo], summaries: Dict[str, str], minio_base_url: str, minio_bucket_name: str) -> str:
        """
        上传文件图片到minio，并且更新md中的图片地址以及摘要
        Args:
            object_dir_name: Minio对象存储的目录名
            md_content: md内容
            img_info_list: 所有图片信息
            summaries: 图片摘要信息
            minio_base_url: Minio的地址
            minio_bucket_name: Minio的桶名

        Returns:
            str: 更新后的md内容

        """

        # 1. 上传
        remote_urls = self._upload_all(object_dir_name, img_info_list, minio_base_url, minio_bucket_name)

        # 2. 更新
        md_content = self._update_md(md_content, summaries, remote_urls)
        return md_content

    def _upload_all(self, object_dir_name: str, img_info_list: List[ImageInfo], minio_base_url: str, minio_bucket_name: str) -> Dict[str, str]:

        remote_urls = {} # 存储图片名和远程地址的字典
        # 1. 获取Minio客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            self.logger.error(f"获取Minio客户端失败: {e}")
            for img_info in img_info_list:
                remote_urls[img_info.name] = img_info.path # 兼容，返回本地路径
            return remote_urls

        # 2. 遍历上传每一个图片
        for img_info in img_info_list:
            object_name = f"{object_dir_name}/{img_info.name}" # Minio对象存储的路径
            try:
                # 2.1 上传图片到Minio
                minio_client.fput_object(minio_bucket_name, object_name, img_info.path)
                # 2.2 拼装远程地址
                # http://192.168.64.150:9000/桶名/对象名
                remote_urls[img_info.name] = f"{minio_base_url}/{minio_bucket_name}/{object_name}"
            except Exception as e:
                self.logger.error(f"上传图片 {img_info.path} 到Minio失败: {e}")
                remote_urls[img_info.name] = img_info.path # 兼容，返回本地路径
                continue
        self.logger.info(f"获取到远程图片地址: {remote_urls}")
        return remote_urls

    def _update_md(self, md_content: str, summaries: Dict[str, str], remote_urls: Dict[str, str]) -> str:
        """
        更新MD中的图片描述和远程图片地址
        Args:
            md_content: MD内容
            summaries: vlm生成的图片摘要信息
            remote_urls: minio上传后的远程图片地址

        Returns:
            更新后的MD内容

        """
        # 利用正则寻找(捕获组：()一个捕获组：group(0): 存储整个匹配到的内容 group(1): 图片的摘要 group(2): 图片地址)
        pattern = re.compile(r'!\[(.*?)\]\((.*?)\)') # 匹配 ![图片描述](图片地址) 的正则表达式，(.*?) 表示捕获匹配了括号内表达式要求的内容

        def replacer(match: re.Match) -> str:
            """

            Args:
                match:

            Returns:
                ![摘要](远程图片地址)
            """
            origin_img_path = match.group(2)  # MD内容中匹配到正则表达式的图片地址
            img_name_in_md = Path(origin_img_path).name
            for img_name, img_summary in summaries.items():

                if img_name == img_name_in_md:
                    return f"![{img_summary}]({remote_urls[img_name_in_md]})"
            return match.group(0)  # 如果没有匹配到，返回原来的内容

        return pattern.sub(replacer, md_content)  # 替换所有匹配到的内容


class MarkDownToImgNode(BaseNode):
    """
    主要职责：
    1. 得到四个类的实例对象
    2. 分别调用四个实例对象的处理方法
    """

    name = "md_to_img_node"

    def __init__(self):
        super().__init__()  # 显示调用父类的构造方法
        self._md_file_handler = _MdFileHandler(self.logger, node_name=self.name)
        self._image_scanner = _ImageScanner(self.logger)
        self._vlm_summarizer = _VLMSummarizer(self.logger, self.config.requests_per_minute)
        self._image_uploader = _ImageUploader(self.logger)

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        入口逻辑
        :param state:
        :return:
        """

        config = self.config

        # 1. 操作md_file_handler
        self.log_step("step1", "读取MD内容、路径以及图片的目录")
        md_content, md_path_obj, img_dir_obj = self._md_file_handler.validate_and_read_md(state)
        # 1.1 判断图片目录不存在
        if not img_dir_obj.exists():
            state["md_content"] = md_content
            return state

        # 2. 操作image_scanner
        self.log_step("step2", "开始扫描图片目录")
        img_info_list: List[ImageInfo] = self._image_scanner.scan_imgs_dir(img_dir_obj,
                                                                           md_content,
                                                                           md_path_obj,
                                                                           config.image_extensions,
                                                                           config.img_content_length)

        # 3. 操作vlm_summarizer
        self.log_step("step3", "开始调用VLM模型生成图片摘要")
        summaries: Dict[str, str] = self._vlm_summarizer.summary_all(md_path_obj.stem, img_info_list, config.vl_model)

        # 4. 操作image_uploader
        self.log_step("step4", "开始上传图片到Minio")
        new_md_content = self._image_uploader.upload_and_replace(md_path_obj.stem, md_content, img_info_list, summaries, config.get_minio_base_url(), config.minio_bucket)

        # 5. 备份新的md文件
        self.log_step("step5", "备份新的MD文件")
        self._md_file_handler.backup(md_path_obj, new_md_content)

        state["md_content"] = new_md_content

        return state

if __name__ == "__main__":
    setup_logging()
    md_img_node = MarkDownToImgNode()

    init_state = create_default_state(
        md_path=r"../temp_dir/万用表的使用/markdown.md"
    )

    md_img_node.process(init_state)
