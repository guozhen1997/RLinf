import os
from typing import Dict, List

def generate_config_files(
    template_path: str,
    output_dir: str,
    variations: List[Dict],
    base_filename: str = "config"
):
    """
    基于模板生成多个不同的配置文件
    
    Args:
        template_path: 模板文件路径
        output_dir: 输出目录
        variations: 配置变化列表，每个元素是一个字典，包含要修改的参数和值
        base_filename: 基础文件名
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取模板内容
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()
    
    # 生成每个变体配置文件
    for i, variation in enumerate(variations):
        content = template_content
        
        # 替换配置参数
        for key, value in variation.items():
            # 处理嵌套参数（如 actor.model.precision）
            key_parts = key.split('.')
            # 构建匹配模式（考虑可能的空格）
            pattern = f"{'.'.join(key_parts)}:.*"
            # 构建替换值
            if isinstance(value, bool):
                replacement = f"{'.'.join(key_parts)}: {str(value).lower()}"
            else:
                replacement = f"{'.'.join(key_parts)}: {value}"
            
            # 替换内容
            import re
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        # 保存文件
        filename = f"{base_filename}_{i}.yaml"
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"已生成配置文件: {output_path}")

if __name__ == "__main__":
    # 模板文件路径（请替换为您的实际模板路径）
    TEMPLATE_PATH = "./config/robotwin_ppo_openvlaoft_eval.yaml"
    
    # 输出目录
    OUTPUT_DIR = "./config"
    
    # 定义要生成的配置变体
    # 每个字典代表一个变体，键是要修改的参数路径，值是新值
    CONFIG_VARIATIONS = [
        # 变体1: 修改学习率和batch size
        {
            "defaults.env/eval": "robotwin_beat_block_hammer",
        },
        # # 变体2: 修改温度参数和clip比率
        # {
        #     "algorithm.sampling_params.temperature_train": 2.0,
        #     "algorithm.sampling_params.temperature_eval": 2.0,
        #     "algorithm.clip_ratio_high": 0.3,
        #     "algorithm.clip_ratio_low": 0.25,
        #     "runner.experiment_name": "test_openvla_temp2.0"
        # },
    ]
    
    # 生成配置文件
    generate_config_files(
        template_path=TEMPLATE_PATH,
        output_dir=OUTPUT_DIR,
        variations=CONFIG_VARIATIONS,
        base_filename="beat_block_hammer"
    )