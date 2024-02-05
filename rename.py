import re
import os

def rename_lab_files(directory):
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".lab"):
            # 使用正则表达式匹配文件名中的整数部分
            match = re.search(r"(\d+)", filename)
            if match:
                # 提取整数部分并进行加法运算
                number = int(match.group(1))
                new_number = number + 147

                # 构建新的文件名
                new_filename = re.sub(r"\d+", str(new_number), filename)

                # 重命名文件
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)
                os.rename(old_path, new_path)
                print(f"重命名文件: {filename} -> {new_filename}")

# 指定包含.lab文件的目录路径
directory_path = "/home/luyao/project/Bert-VITS2/raw/luyao_02"

# 调用函数进行重命名
rename_lab_files(directory_path)