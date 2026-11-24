from pathlib import Path

from knowledge.processor.import_processor.base import BaseNode, setup_logging, T
from knowledge.processor.import_processor.state import ImportGraphState, create_default_state
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError


class EntryNode(BaseNode):
    name = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        根据上传文件的后缀 修改state中的is_md_read_enabled 和 is_pdf_read_enabled
        Args:
            state:

        Returns:

        """

        # 1. 获取上传的文件
        import_file_path = state.get("import_file_path", "")
        file_dir = state.get("file_dir", "")

        # 2. 判断
        if not import_file_path:
            raise StateFieldError(node_name=self.name, field_name="import_file_path", expected_type=str)

        if not file_dir:
            raise StateFieldError(node_name=self.name, field_name="file_dir", expected_type=str)

        # 3. Path标准化
        import_file_path_obj = Path(import_file_path)
        file_dir_obj = Path(file_dir)

        # 4. 判断
        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.name, field_name="import_file_path", expected_type=Path, message=f"文件不存在: {import_file_path}")
        if not file_dir_obj.exists():
            raise StateFieldError(node_name=self.name, field_name="file_dir", expected_type=Path, message=f"文件夹不存在: {file_dir}")

        # 5. 获取文件的后缀
        if import_file_path_obj.suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = str(import_file_path_obj)
        elif import_file_path_obj.suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = str(import_file_path_obj)
        else:
            self.logger.error("不支持的文件类型: %s", import_file_path_obj.suffix)
            raise ValidationError(node_name=self.name, message=f"不支持的文件类型: {import_file_path_obj.suffix}")

        # 6. 获取上传文件的标题，更新到state中
        state["file_title"] = import_file_path_obj.stem

        # 7. 返回state
        return state

if __name__ == '__main__':
    entry_node = EntryNode()

    init_state = create_default_state(
        import_file_path=r"../temp_dir/万用表的使用.pdf",  # r 表示原始字符串，避免转义字符
        file_dir=r"../temp_dir/"
    )
    result = entry_node(init_state)

    import json

    print(json.dumps(result, ensure_ascii=False, indent=4))
