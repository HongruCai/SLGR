from transformers import T5Config, T5ForConditionalGeneration
from transformers import LlamaConfig, LlamaForCausalLM

def load_and_count(source_file, target_file):
    # 打开并读取 source 和 target 文件
    with open(source_file, 'r', encoding='utf-8') as src, open(target_file, 'r', encoding='utf-8') as tgt:
        inputs = src.readlines()
        labels = tgt.readlines()
    
    # 检查行数是否匹配
    if len(inputs) != len(labels):
        raise ValueError("The number of lines in source and target files do not match!")
    
    # 输出总行数
    total_lines = len(inputs)
    print(f"Total number of lines: {total_lines}")
    
    return inputs, labels, total_lines
# 示例调用
source_path = "../data/training/llama_train.source"
target_path = "../data/training/llama_train.target"
inputs, labels, total_lines = load_and_count(source_path, target_path)

