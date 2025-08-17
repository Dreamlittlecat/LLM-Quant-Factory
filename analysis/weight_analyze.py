import torch
import os
import json
import math
import argparse
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM
from modelutils import find_layers

def compute_weight_linear_ratio_avg(data: dict) -> dict:
    """
    输入为每层的详细指标数据，输出为每个投影层的 std、mup 和 ratio 的平均值。
    """
    proj_stats = defaultdict(lambda: {"std": [], "mup": [], "ratio": []})

    for layer_data in data.values():
        for proj_name, metrics in layer_data.items():
            if proj_name in ["block_std", "block_mup"]:
                continue
            proj_stats[proj_name]["std"].append(metrics["std"])
            proj_stats[proj_name]["mup"].append(metrics["mup"])
            proj_stats[proj_name]["ratio"].append(metrics["ratio"])

    avg_results = {}
    for proj_name, metrics in proj_stats.items():
        avg_results[proj_name] = {
            "std": sum(metrics["std"]) / len(metrics["std"]),
            "mup": sum(metrics["mup"]) / len(metrics["mup"]),
            "ratio": sum(metrics["ratio"]) / len(metrics["ratio"]),
        }
    return avg_results

def sort_layers_by_block_metrics(data: dict):
    # 抽取所有层的 layer_id、block_std、block_mup
    layer_metrics = [
        (layer_id, layer_info["block_std"], layer_info["block_mup"])
        for layer_id, layer_info in data.items()
        if "block_std" in layer_info and "block_mup" in layer_info
    ]

    # 按 block_std 升序排序（越小越靠前）
    sorted_by_std = sorted(layer_metrics, key=lambda x: x[1])
    # 按 block_mup 降序排序（越大越靠前）
    sorted_by_mup = sorted(layer_metrics, key=lambda x: -x[2])

    # 返回 layer_id 和对应的 block_std / block_mup 值
    sorted_std_layers = [{layer_id: block_std} for layer_id, block_std, _ in sorted_by_std]
    sorted_mup_layers = [{layer_id: block_mup} for layer_id, _, block_mup in sorted_by_mup]

    return {
        "sorted_std_layers": sorted_std_layers,
        "sorted_mup_layers": sorted_mup_layers
    }


def mup_incoherent(weight_tensor):
    """计算 MUP incoherence 指标"""
    max_weight = torch.max(weight_tensor)
    f_norm = torch.norm(weight_tensor)
    return max_weight * math.sqrt(weight_tensor.numel()) / f_norm

def get_model(model_path):
    """加载模型"""
    print("Loading model:", model_path)

    # 跳过权重初始化
    def skip(*args, **kwargs): pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    model.seqlen = getattr(model.config, "max_position_embeddings", 2048)
    return model

def model_weight_analysis(model_path):
    """分析模型权重，计算 std、mup_incoherent 等指标"""
    model = get_model(model_path)

    if hasattr(model.model, "decoder"):
        layers = model.model.decoder.layers  # e.g., OPT
    elif hasattr(model.model, "layers"):
        layers = model.model.layers  # e.g., LLaMA
    else:
        raise ValueError("Unsupported model architecture.")

    print(f"Model has {len(layers)} layers.")

    block_std = []
    block_mup = []
    merged_layer_info = {}

    for i, layer in enumerate(tqdm(layers, desc="Analyzing Layers")):
        sub_layers = find_layers(layer)
        total_params = torch.tensor([])

        std_dict = {}
        mup_dict = {}
        ratio_dict = {}

        for name, module in sub_layers.items():
            weight = module.weight.data
            std_dict[name] = torch.std(weight).item()
            mup_dict[name] = mup_incoherent(weight).item()
            total_params = torch.cat([total_params, weight.view(-1)])

        total_std = torch.std(total_params).item()
        total_mup = mup_incoherent(total_params).item()

        for name, module in sub_layers.items():
            ratio_dict[name] = module.weight.data.numel() / total_params.numel()

        layer_id = str(i)
        merged_layer_info[layer_id] = {
            "block_std": total_std,
            "block_mup": total_mup
        }
        for name in std_dict:
            merged_layer_info[layer_id][name] = {
                "std": std_dict[name],
                "mup": mup_dict[name],
                "ratio": ratio_dict[name]
            }

        del total_params

    return merged_layer_info

def save_analysis_results(output_dir, merged_layer_info):
    """保存分析结果到 JSON 文件"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")

    with open(os.path.join(output_dir, "linear_info.json"), "w") as f:
        json.dump(merged_layer_info, f, indent=4)

    avg_results = compute_weight_linear_ratio_avg(merged_layer_info)
    with open(os.path.join(output_dir, "linear_avg.json"), "w") as f:
        json.dump(avg_results, f, indent=4)
    sorted_layers = sort_layers_by_block_metrics(merged_layer_info)
    with open(os.path.join(output_dir, "sorted_layers.json"), "w") as f:
        json.dump(sorted_layers, f, indent=4)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data/xjh/model_weight/llama/llama3-8b")
    return parser.parse_args()

def main():
    args = get_args()
    model_path = args.model_path
    output_dir = f"./output/{os.path.basename(model_path)}"

    results = model_weight_analysis(model_path)
    save_analysis_results(output_dir, results)

if __name__ == "__main__":
    main()