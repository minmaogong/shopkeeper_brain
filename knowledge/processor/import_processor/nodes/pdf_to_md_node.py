from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.state import ImportGraphState

class PdfToMdNode(BaseNode):

    name = 'PdfToMd'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点的处理逻辑入口
        :param state:
        :return:
        """

        # 核心逻辑（接收pdf文件path，利用minerU解析工具将pdf解析成md）



