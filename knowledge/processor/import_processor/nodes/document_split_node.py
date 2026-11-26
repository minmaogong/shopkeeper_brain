import re
from typing import Tuple, List, Dict, Any


from knowledge.processor.import_processor.base import BaseNode, setup_logging, T
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_processor.config import ImportConfig, get_config

class DocumentSplitNode(BaseNode):
    name = 'document_split_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        文档切分的核心逻辑入口
        Args:
            state:

        Returns:

        """
        # 1. 参数校验
        md_content, file_title, max_content_length, min_content_length = self._validate_state(state, get_config())

        # 2. 切分（一级策略：根据md文档中的标题来切分）多个章节（章节：标题之间的内容）
        sections: List[Dict[str, Any]] = self._split_by_headings(md_content, file_title)

    def _validate_state(self, state: ImportGraphState, config: ImportConfig) -> Tuple[str, str, int, int]:
        self.log_step("step1", "切分文档的参数校验以及获取...")

        # 1. 获取md_content
        md_content = state.get("md_content")

        # 2. 统一换行符
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 3. 获取文件标题
        file_title = state.get("file_title")

        # 4. 校验最大最小值
        if config.max_content_length <= 0 or config.min_content_length <= 0 or config.max_content_length < config.min_content_length:
            raise ValidationError(node_name=self.name, message=f"切片长度参数校验失败")

        return md_content, file_title, config.max_content_length, config.min_content_length

    def _split_by_headings(self, md_content: str, file_title: str) -> List[Dict[str, Any]]:
        """
        parent_title: 封装的原因主要是为了后面短section在合并的时候有一个判断标准（同源：同一个父标题）
        根据标题来切分（# {1,6}都有可能）
        Args:
            md_content: 切分的md
            file_title: 上传文档标题

        Returns:
            List[Dict[str, Any]]: 切分后的多个章节
        """
        in_fence = False # 是否在代码块内
        body_lines = []
        sections: List[Dict[str, Any]] = [] # 最终收集到的章节列表
        current_title = ""
        hierarchy = [""] * 7 # 标题层级栈，用于追踪当前标题的层级关系 长度为7，第一个（0）不用
        current_level = 0

        def _flush() -> List[Dict[str, Any]]:
            """
            打包section
            {
            "body": "收集到的所有行",
            "title": "当前内容的标题",
            "parent_title": "当前内容的父标题（最麻烦）",
            "file_title": "文档的标题（最简单）"
            }
            Returns:

            """

            # 1. 处理内容行
            body="\n".join(body_lines).strip()
            if current_title or body:
                parent_title = ""
                for i in range(current_level-1, 0, -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break

                if not parent_title:
                    parent_title = current_title if current_title else file_title

                sections.append({
                    "body": body,
                    "title": current_title if current_title else file_title, # 当前内容的标题
                    "parent_title": parent_title,  # 需要根据标题层级来确定父标题
                    "file_title": file_title
                })


        # 1. 根据\n切分md_content
        md_lines = md_content.split("\n")
        # 2. 定义正则（正则的规则是从MD中找标题#{1,6 }） . 表示任意字符； * 表示匹配零个或多个前面的字符； + 表示匹配至少一个前面的字符； ():捕获组：两个() 意味着产生3个groups，group(0)是整个匹配的字符串，group(1)是第一个()匹配的内容 #(1)#(6)，group(2)是第二个()匹配的内容 标题的内容。
        heading_re = re.compile(r'^\s*(#{1,6})\s+(.+)') # ^\s 表示匹配行首的空白字符，即要求标题前面可以有空格，但必须在行首。 ^\s* 表示匹配行首的零个或多个空白字符。 (#{1,6}) 表示匹配一个到六个连续的 # 字符，表示 Markdown 中的标题级别。 \s+ 表示匹配一个或多个空白字符，确保标题文本与 # 之间有至少一个空格。 (.+) 表示匹配标题文本，即标题内容，可以是任意字符，至少一个字符。

        # 3. 遍历切分后的md_lines
        for md_line in md_lines:
            # 3.1 检测代码块围栏边界（``` 或 ~~~）
            if md_line.strip().startswith("```") or md_line.strip().startswith("~~~"):
                in_fence = not in_fence

            # 3.2 判断是否为标题（不在代码块内时）
            match = heading_re.match(md_line) if not in_fence else None

            # 3.3 判断是否匹配到标题
            if match: # 匹配到标题
                # 将body_lines中收集到的行封装到section中
                _flush()

                current_title = md_line # md_line = match.group(2) # 当前标题的内容
                level = len(match.group(1)) # 当前标题的层级 （# {1,6}）
                current_level = level
                hierarchy[level] = current_title # 更新当前层级的标题

                for i in range(level+1, 7): # 清空当前层级以下的标题
                    hierarchy[i] = ""

            else: # 非标题或者代码块的内容
                body_lines.append(md_line)

        _flush()

        return sections






