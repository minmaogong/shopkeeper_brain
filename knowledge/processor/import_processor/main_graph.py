import json
import logging

from langgraph.graph.state import StateGraph, CompiledStateGraph
from langgraph.graph import END

from knowledge.processor.import_processor.base import setup_logging
from knowledge.processor.import_processor.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_processor.nodes.md_to_img_node import MarkDownToImgNode
from knowledge.processor.import_processor.nodes.entry_node import EntryNode
from knowledge.processor.import_processor.state import ImportGraphState, create_default_state

"""
编排节点

定义节点
定义条件边
定义顺序边
运行整个pineline图谱的各个节点

"""

def import_router(state: ImportGraphState) -> str:
    """
    根据state中的is_md_read_enabled 和 is_pdf_read_enabled 来决定下一步的节点
    Returns:

    """
    # 1. 获取上传的文件属于pdf or md
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md_node"

    if state.get("is_md_read_enabled"):
        return "md_to_img_node"

    return END


def import_graph() -> CompiledStateGraph:
    """
    职责：
    1. 定义运行时图状态workflow
    2. 定义节点 （入口节点、普通业务节点）
    3. 定义边（条件边、普通业务边）
    4. 返回运行时的状态
    Returns:

    """

    # 1. 定义运行时图状态workflow
    work_flow = StateGraph(state_schema=ImportGraphState) # type: ignore

    # 2. 定义入口节点
    work_flow.set_entry_point("entry_node")

    # 3. 定义其他节点名和节点实例的映射表
    node_name_obj = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PdfToMdNode(),
        "md_to_img_node": MarkDownToImgNode(),
    }

    # 4. 遍历映射表添加
    for node_name, node_obj in node_name_obj.items():
        work_flow.add_node(node_name, node_obj)

    # 5. 定义边
    # 5.1 定义条件边 source   path   path_map
    work_flow.add_conditional_edges("entry_node", path=import_router, path_map={
        "pdf_to_md_node": "pdf_to_md_node", # key: 路由函数的返回值， value: 节点的名字
        "md_to_img_node": "md_to_img_node",
        END: END
    })

    # 5.2 定义业务边
    work_flow.add_edge("pdf_to_md_node", "md_to_img_node")
    work_flow.add_edge("md_to_img_node", END)

    # 5.3 编译
    compiled_state_graph = work_flow.compile()

    # 5.4 返回编译后的图谱
    return compiled_state_graph

import_app = import_graph()

##########################################
# 测试
##########################################
def run_import_graph():
    # 1. 定义运行graph流程的祖杭台
    graph_state = create_default_state(
        import_file_path=r"./temp_dir/万用表的使用.pdf",  # r 表示原始字符串，避免转义字符
        file_dir=r"./temp_dir/"
    )

    # stream: 迭代整个graph图状态可以得到每一个节点的事件(节点的名字以及节点操作完state之后的新状态)
    final_state = {}
    for event in import_app.stream(graph_state):
        for node_name, state in event.items():
            print(f"当前正在执行的节点： {node_name}")
            print(f"当前正在执行的节点状态: {state}")
            final_state = state
    return final_state

if __name__ == '__main__':
    setup_logging(level=logging.INFO)
    final_state = run_import_graph()
    print(json.dumps(final_state, ensure_ascii=False, indent=4))

    # 整个执行的状态图(方便观测) ascii
    print(import_app.get_graph().print_ascii())
