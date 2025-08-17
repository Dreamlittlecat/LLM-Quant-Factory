import json
import argparse
import os

def round_to_nearest_quarter(x):
    return round(x * 4) / 4

def allocate_osr_incremental(data: dict, avg_osr: float, alpha: float = 2.0, base_osr: float = 0.5):
    print(f"avg_osr:{avg_osr},alpha{alpha},base_osr:{base_osr}")
    layer_std = [(layer_id, layer_info["block_std"]) for layer_id, layer_info in data.items()]
    num_layers = len(layer_std)

    # 基础分配部分
    base_total = base_osr * num_layers
    remaining_total = avg_osr * num_layers - base_total
    if remaining_total <= 0:
        print("⚠️ Warning: base_osr 已经超过或等于目标 mean_osr,无法再分配增量部分")
        remaining_total = 0.0

    # 计算重要性分数（用于增量）
    scores = [(layer_id, 1.0 / (std ** alpha)) for layer_id, std in layer_std]
    total_score = sum(score for _, score in scores)

    # 分配增量 + 基础
    raw_osr = {
        layer_id: base_osr + (score / total_score) * remaining_total
        for layer_id, score in scores
    }

    # 四分之一舍入
    rounded_osr = {
        layer_id: round_to_nearest_quarter(osr)
        for layer_id, osr in raw_osr.items()
    }

    # 调整因子（尽量逼近目标 OSR）
    actual_mean = sum(rounded_osr.values()) / num_layers
    scale_factor = avg_osr / actual_mean if actual_mean > 0 else 1.0

    final_osr = {
        layer_id: round_to_nearest_quarter(osr * scale_factor)
        for layer_id, osr in rounded_osr.items()
    }

    return final_osr

def compute_incremental_osr(config_dict, target_mean_osr, base_osr=1.0, alpha=1.0):
    # Step 1: 基础分配部分
    base_contrib_total = sum(
        item['ratio'] * base_osr 
        for item in config_dict.values() 
        if isinstance(item, dict)
    )

    # Step 2: 计算剩余需要分配的部分
    target_total = target_mean_osr
    remaining_budget = target_total - base_contrib_total

    if remaining_budget <= 0:
        print("Warning: base_osr 已经超过或接近目标 mean_osr")
        remaining_budget = 0.0

    # Step 3: 按照 std 的倒数权重分配剩余部分
    weighted_inverse_std_sum = sum(
        item['ratio'] / (item['std'] ** alpha) 
        for item in config_dict.values() 
        if isinstance(item, dict)
    )
    k = remaining_budget / weighted_inverse_std_sum if weighted_inverse_std_sum > 0 else 0.0

    result = {}
    for name, item in config_dict.items():
        if not isinstance(item, dict):
            continue
        std_weighted = item['std'] ** alpha
        incremental_osr = k / std_weighted if std_weighted > 0 else 0.0
        total_osr = base_osr + incremental_osr
        rounded_osr = round_to_nearest_quarter(total_osr)
        result[name] = {
            "ratio": item['ratio'],
            "std": item['std'],
            "base_osr": base_osr,
            "incremental_osr": incremental_osr,
            "computed_osr_proj": rounded_osr,
            "mean_osr_contrib": item['ratio'] * rounded_osr
        }
    return result

def generate_config(file_path, output_dir, target_avg_osr=2, alpha=1.5, base_osr=1, verbose=False):
    """
    Generate OSR configuration based on the input linear info file.
    
    Args:
        file_path: Path to the input linear info JSON file
        output_dir: Directory to save the output configuration
        target_avg_osr: Target average OSR (default: 2)
        alpha: Alpha parameter for weighting (default: 1.5)
        base_osr: Base OSR for each layer (default: 1)
        verbose: Whether to print detailed information (default: False)
    
    Returns:
        Path to the generated configuration file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration data
    print(f"Loading linear info from: {file_path}")
    with open(file_path, 'r') as f:
        config = json.load(f)
    
    # Compute block-level OSR allocation
    print(f"Computing block-level OSR allocation with target_avg_osr={target_avg_osr}, alpha={alpha}, base_osr={base_osr}")
    result_block = allocate_osr_incremental(config, avg_osr=target_avg_osr, alpha=alpha, base_osr=base_osr)
    
    # Output block-level results if verbose
    if verbose:
        total_contrib = 0
        for name, osr_val in result_block.items():
            print(f"{name}: osr_proj={osr_val:.2f}")
            total_contrib += osr_val
        
        actual_mean_osr = total_contrib / len(result_block)
        print(f"\nTarget Mean OSR: {target_avg_osr:.4f}, Actual Mean OSR (after rounding): {actual_mean_osr:.4f}")
    
    # Compute projection-level OSR allocation
    print("Computing projection-level OSR allocation")
    result = {}
    for layer_id, item in config.items():
        target_mean_osr = result_block[layer_id]
        
        # Adjust alpha based on target OSR
        layer_alpha = 0.1 if target_mean_osr >= 3.5 else 1.5
        
        result_linear = compute_incremental_osr(
            item, 
            target_mean_osr=target_mean_osr, 
            base_osr=base_osr, 
            alpha=layer_alpha
        )
        
        # Store results
        result[layer_id] = {}
        for name, value in result_linear.items():
            if verbose:
                print(f"  {layer_id}.{name}: osr_proj={value['computed_osr_proj']:.2f}, base={value['base_osr']:.2f}, "
                      f"inc={value['incremental_osr']:.2f}, contrib={value['mean_osr_contrib']:.4f}")
            
            result[layer_id][name] = {
                "osr_proj": value['computed_osr_proj'],
            }
    
    # Generate output filename
    model_name = os.path.basename(file_path).split('_')[0] if '_' in os.path.basename(file_path) else "model"
    output_file = os.path.join(output_dir, f"{model_name}_osr_{target_avg_osr:.2f}.json")
    
    # Save result
    print(f"Saving configuration to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Generate OSR configuration for model quantization")
    parser.add_argument("--file_path", type=str, required=True,
                        help="Path to the input linear info JSON file")
    parser.add_argument("--output_dir", type=str, default="./configs",
                        help="Directory to save the output configuration")
    parser.add_argument("--target_osr", type=float, default=2.00,
                        help="Target average OSR value")
    parser.add_argument("--alpha", type=float, default=1.5,
                        help="Alpha parameter for weighting")
    parser.add_argument("--base_osr", type=float, default=1.0,
                        help="Base OSR value for each layer")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    
    args = parser.parse_args()
    
    output_file = generate_config(
        file_path=args.file_path,
        output_dir=args.output_dir,
        target_avg_osr=args.target_osr,
        alpha=args.alpha,
        base_osr=args.base_osr,
        verbose=args.verbose
    )
    
    print(f"Configuration generation complete: {output_file}")

if __name__ == "__main__":
    main()