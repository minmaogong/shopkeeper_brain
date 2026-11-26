import re
from typing import Tuple, List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

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

        # 3. 二次切分或者合并
        self._split_and_merge(sections, max_content_length, min_content_length)

        # 4. 组装成chunk对象

        # 5. 备份（观察）

        # 6. 更新state (chunks)

        # 7. 返回state
        return state

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

                body_lines = [] # 清空body_lines，准备收集新的内容

            else: # 非标题或者代码块的内容
                body_lines.append(md_line)

        _flush()

        return sections

    def _split_and_merge(self, sections: List[Dict[str, Any]], max_content_length: int, min_content_length: int) -> List[Dict[str, Any]]:
        """
        切分较大章节（section）以及合并较小章节（section）
        Args:
            sections: 所有经过一级切分后的章节
            max_content_length: 最大内容长度（content：title+\n\n+body）如果section中的内容长度超过阈值max_content_length，就需要进行切割，反之不需要切割（）尽量保证不要太多的section进行二次切割
            min_content_length: 最小内容长度：如果section中的内容长度不足阈值min_content_length，就需要对该section进行合并，同源机制合并（section相同的父标题菜合。尽量保证确实比较小的内容才合并，不要把大多数的section都合并了）

        Returns:
            先切后合

        """

        # 1. 切分
        current_sections = []
        for section in sections:
            current_sections.extend(self._split_long_section(section, max_content_length)) # extend() 方法用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。与 append() 方法不同，append() 方法只能添加一个元素，而 extend() 方法可以添加多个元素。

        # 2. 合并
        final_sections = self._merge_short_section(current_sections, min_content_length)

        return final_sections

    def _split_long_section(self, section: Dict[str, Any], max_content_length: int) -> List[Dict[str, Any]]:
        """
        切分长的章节内容
        Args:
            section: 当前章节
            max_content_length: 最大长度阈值

        Returns:
            List[Dict[str, Any]]: 切分后的章节列表

        """

        # 1. 获取section对象的属性
        body = section.get("body") # 行内容
        title = section.get("title") # 标题
        parent_title = section.get("parent_title") # 父标题
        file_title = section.get("file_title") # 文件标题

        # 2. 判断当前section的内容长度是否超过阈值
        title_prefix = f"{title}\n\n"
        total_length = len(title_prefix) + len(body)
        if total_length <= max_content_length:
            return [section]

        # 3. 使用递归切分器切分body内容 pip install langchain_text_splitters
        # 3.1 切分器对象 # chunk_size: chunk块的大小 chunk_overlap: 块与块之间的重叠部分的大小 separators: 分割符列表 【"\n\n", "\n", " ", ""】keep_separator: 是否保留分隔符
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=, chunk_overlap=, separators=, keep_separator=True)

        # 3.2 切分器对象切分




if __name__ == '__main__':

    document_split_node = DocumentSplitNode()

    md_path = r"../temp_dir/万用表的使用/markdown_new.md"

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    init_state = {
        "file_title": "万用表的使用",
        "md_content": md_content,
    }
    document_split_node.process(init_state)




