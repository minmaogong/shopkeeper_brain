# 1. 内置包
import subprocess, json, time
from typing import Tuple
from pathlib import Path

# 2. 三方包

# 3. 自己的包
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError, PdfConversionError
from knowledge.processor.import_processor.state import create_default_state

class PdfToMdNode(BaseNode):

    name = 'pdf_to_md_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点的处理逻辑入口
        :param state:
        :return:
        """

        # 核心逻辑（接收pdf文件path，利用minerU解析工具将pdf解析成md）

        # 1. 获取导入文件的路径以及输出目录
        import_file_path_obj, file_dir_obj = self._invalid_state(state)

        output_zip_path_obj = file_dir_obj / f"{import_file_path_obj.stem}.zip" # import_file_path_obj.stem 获取不带后缀的文件名
        # 2. 执行mineru解析（命令：mineru-kit parse input_file_path -o output_zip_path --tier standard --format zip --source=local）
        processed_code = self._execute_mineru_parse(import_file_path_obj, output_zip_path_obj)

        if processed_code != 0:
            raise PdfConversionError(node_name=self.name, message=f"MinerUPDF解析失败，状态码: {processed_code}")

        # 3. 解压zip文件，获取md文件路径
        md_path = self.unzip_get_md_path(output_zip_path_obj)

        # 4. 更新状态state["md_path"]
        state["md_path"] = md_path

        # 5. 返回更新后的状态
        return state

    def _invalid_state(self, state: ImportGraphState) -> Tuple[Path, Path]:
        """

        :param state: 导入图谱节点状态
        :return: 导入文件的路径以及输出目录
        Tuple[Path, Path]
        """

        self.log_step("step1", "准备校验和获取解析文件路径和输出目录")

        # 1. 获取解析的文件path
        import_file_path = state.get("import_file_path", "")

        # 2. 判断是否为空
        if not import_file_path:
            raise StateFieldError(node_name=self.name, field_name="import_file_path", expected_type=str)

        # 3. 标准化解析文件的路径
        import_file_path_obj = Path(import_file_path)

        # 4. 判断是否是一个有效的路径
        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.name, field_name="import_file_path", expected_type=str
                                , message=f"导入文件路径不存在: {import_file_path}")

        # 5. 输出文件目录
        file_dir = state.get("file_dir", "")

        # 6. 判断是否为空
        if not file_dir:
            file_dir = import_file_path_obj.parent

        # 7. 判断输出文件目录是否存在
        if not Path(file_dir).exists():
            raise StateFieldError(node_name=self.name, field_name="file_dir", expected_type=str
                                , message=f"输出文件目录不存在: {file_dir}")

        # 8. 标准化输出文件目录
        file_dir_obj = Path(file_dir)

        self.logger.info(f"解析文件路径: {import_file_path_obj}, 输出文件目录: {file_dir_obj}")

        # 9. 返回解析文件路径和输出目录
        return import_file_path_obj, file_dir_obj

    def _execute_mineru_parse(self, import_file_path_obj: Path, output_zip_path_obj: Path) -> int:
        """
        执行mineru解析命令
        :param import_file_path_obj: 解析文件的路径
        :param output_zip_path_obj: 解析后的输出zip文件路径
        :return: 状态[0或者非0]
        0: 成功（底层封装的状态码）
        非0: 失败
        """
        # mineru-kit parse input_file_path -o output_zip_path --tier standard --format zip --source=local
        # 1. 子进程执行cmd命令

        # 1. 定义cmd
        cmd=[
            "mineru-kit",
            "parse",
            str(import_file_path_obj),
            "-o",
            str(output_zip_path_obj),
            "--tier",
            "standard",
            "--format",
            "zip"
        ]

        # 2. 子进程执行cmd命令(子进程解析的日志【正常日志和错误日志】)
        start_time = time.time()
        proc = subprocess.Popen(args=cmd,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, # 将标准错误输出重定向到标准输出
                         text=True, # 将输出解码为字符串，false则输出为字节流
                         errors="replace", # 遇到无法解码的字符时，使用替换字符替换
                         encoding="utf-8",
                         bufsize=1, # 行缓冲，实时输出日志。遇到换行符时刷新缓冲区
                         universal_newlines=True, # 将输出解码为字符串，false则输出为字
                         )

        # 3. 实时输出日志
        for line in proc.stdout:
            self.logger.info(f"mineru解析日志: {line.strip()}")

        # 4. 等待子进程结束（状态码0：成功 非0: 失败）
        processed_result = proc.wait()
        end_time = time.time()

        if processed_result == 0:
            self.logger.info(f"Mineru解析成功，耗时: {end_time - start_time:.2f}秒")
        else:
            self.logger.info(f"Mineru解析失败，状态码: {processed_result}")

        return processed_result

    def unzip_get_md_path(self, output_zip_path_obj:Path) -> str:
        """
        解压zip文件，获取md文件路径
        :param output_zip_path_obj:
        :return: md文件路径
        """
        # 1. 解压zip文件
        import zipfile

        md_name = output_zip_path_obj.stem

        with zipfile.ZipFile(output_zip_path_obj, 'r') as zip_ref:
            zip_ref.extractall(output_zip_path_obj.parent / md_name) # 解压到导入文件所在目录

        # 2. 获取解压后的md文件路径
        extract_dir = output_zip_path_obj.parent / md_name # 解压后的目录名与导入文件名相同（不带后缀）
        md_file_path_obj = extract_dir / "markdown.md" # 解压后的md文件名为markdown.md

        if not md_file_path_obj.exists():
            raise PdfConversionError(node_name=self.name, message=f"解压后的md文件不存在: {md_file_path_obj}")

        self.logger.info(f"解压后的md文件路径: {md_file_path_obj}")

        return str(md_file_path_obj)


if __name__ == "__main__":
    # 0. 设置日志
    setup_logging()

    # 1. 构建节点示例
    pdf_to_md_node = PdfToMdNode()

    # 2. 构建该节点状态
    init_state= create_default_state(
        import_file_path=r"../temp_dir/万用表的使用.pdf", # r 表示原始字符串，避免转义字符
        file_dir=r"../temp_dir"
    )

    # 3. 调用process方法，传入状态字典
    result = pdf_to_md_node(init_state)

    # 4. 序列化（将对象转成字符串）反序列化（将字符串转成对象）
    result_str = json.dumps(result, indent=4, ensure_ascii=False) # indent=4: 缩进4个空格，ensure_ascii=False: 保留中文字符

    print(result_str)

