import transformers
import torch
import torch.nn as nn
import math
import numpy as np
import sys
import os
import random
import argparse
from transformers import LlamaForCausalLM, OPTForCausalLM, AutoModelForCausalLM
sys.path.append("./sdm_utils")
from sdm_v2 import sdm_quant
from pb_utils import pb_mask
from hadamard import random_hadamard_matrix
import itertools

def zero_ratio(matrix):
    total_elements = matrix.numel()
    zero_elements = torch.sum(matrix == 0).item()
    return zero_elements / total_elements

def find_layers(module, layers=None, name=''):
    if layers is None:
        layers = {}
    for name_curr, module_curr in module.named_children():
        name_new = name + '.' + name_curr if name != '' else name_curr
        if isinstance(module_curr, nn.Linear):
            layers[name_new] = module_curr
        else:
            find_layers(module_curr, layers, name=name_new)
    return layers

def search_scale(weight, OSR, first2):
    hadamard = random_hadamard_matrix(weight.shape[-1], device=weight.device).float()
    weight = weight.float().matmul(hadamard)

    # 高粒度参数网格
    scale_list = [round(x, 2) for x in torch.arange(0.75, 1.01, 0.05).tolist()]
    percentile_scale_list = [round(x, 2) for x in torch.arange(0.75, 0.99, 0.025).tolist()]
    percentile_cut_list = [round(x, 2) for x in torch.arange(0.35, 0.601, 0.05).tolist()]

    best_mse = float('inf')
    best_params = None
    #best_weight_q = None
    #total_combinations = len(scale_list) * len(percentile_scale_list) * len(percentile_cut_list)
    count = 0

    for scale, percentile_scale, percentile_cut in itertools.product(scale_list, percentile_scale_list, percentile_cut_list):
        if percentile_cut > percentile_scale:
            continue
        count += 1

        weight_q = sdm_quant(
            weight.clone(), OSR=OSR,
            scale=scale,
            percentile_scale=percentile_scale,
            percentile_cut=percentile_cut,
            first2=first2
        )
        weight_q = weight_q @ hadamard.T
        mse = torch.mean((weight_q - weight@hadamard.T) ** 2).item()
        #print(mse,best_mse)
        if mse < best_mse:
            best_mse = mse
            best_params = (scale, percentile_scale, percentile_cut)
            #best_weight_q = weight_q

    #print(f"\n✅ [Best Params] scale={best_params[0]}, percentile_scale={best_params[1]}, percentile_cut={best_params[2]}, MSE={best_mse:.6f}")
    return best_params

def get_model_type(model_path):
    """Determine if the model is Llama or OPT based on the path or model files"""
    if "llama" in model_path.lower() or "Llama" in model_path:
        return "llama"
    elif "opt" in model_path.lower() or "OPT" in model_path:
        return "opt"
    else:
        # Try to load config to determine the model type
        try:
            config = transformers.AutoConfig.from_pretrained(model_path)
            if "llama" in config.model_type.lower():
                return "llama"
            elif "opt" in config.model_type.lower():
                return "opt"
        except:
            pass
    return "unknown"

#v1
# def find_target_layers(model, model_type):
#     """Find all target layers based on model type"""
#     all_layers = find_layers(model)
#     target_layers = {}
    
#     for name, layer in all_layers.items():
#         if model_type == "llama" and "down_proj" in name:
#             target_layers[name] = layer
#         elif model_type == "opt" and "fc2" in name:
#             target_layers[name] = layer
    
#     return target_layers


def find_target_layers(model, model_type):
    """Find all target layers based on model type and projection type"""
    all_layers = find_layers(model)
    target_layers = {}
    
    if model_type == "llama":
        # Define the projection types to look for in Llama models
        proj_types = ["down_proj", "q_proj", "v_proj", "o_proj", "k_proj", "up_proj", "gate_proj"]
        for proj_type in proj_types:
            proj_layers = {name: layer for name, layer in all_layers.items() if proj_type in name}
            if proj_layers:
                target_layers[proj_type] = proj_layers
            
    elif model_type == "opt":
        # Define the projection types to look for in OPT models
        proj_types = ["fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj"]
        for proj_type in proj_types:
            proj_layers = {name: layer for name, layer in all_layers.items() if proj_type in name}
            if proj_layers:
                target_layers[proj_type] = proj_layers
    
    return target_layers


def select_random_layers(target_layers, seed=42):
    """Select one random layer for each projection type"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    selected_layers = {}
    
    for proj_type, layers in target_layers.items():
        if layers:  # Make sure there are layers of this type
            layer_name = random.choice(list(layers.keys()))
            selected_layers[proj_type] = {
                "name": layer_name,
                "layer": layers[layer_name]
            }
    
    return selected_layers


def analyze_model_params(model_path, OSR=2, first2=False, device="cpu", groupsize=128, n_sample=16):
    """
    Load model, select a random target layer, and find best scale parameters
    """
    print(f"Loading model from {model_path}")
    model_type = get_model_type(model_path)
    
    if model_type == "unknown":
        print("Could not determine model type. Specify 'llama' or 'opt' explicitly.")
        return None
    print(f"Detected model type: {model_type}")
    
    try:
        if model_type == "llama":
            model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        else:  # opt
            model = OPTForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
            model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None
    
    target_layers = find_target_layers(model, model_type)
    if not target_layers:
        print(f"No appropriate layers found in the model.")
        return None
    
    selected_layers = select_random_layers(target_layers, seed=42)
    results = {}
    
    for proj_type, layer_info in selected_layers.items():
        if model_type == "llama" and proj_type != "down_proj":
            continue
        if model_type == "opt" and proj_type != "fc2":
            continue
        print(f"Processing {proj_type} layer: {layer_info['name']}")
        selected_layer = layer_info['layer']
        
        row_samples = min(groupsize, selected_layer.weight.shape[0])
        col_samples = groupsize * n_sample
        if col_samples > selected_layer.weight.shape[1]:
            col_samples = selected_layer.weight.shape[1]
        
        torch.manual_seed(42)  
        torch.cuda.manual_seed(42)
        row_indices = torch.randperm(selected_layer.weight.shape[0])[:row_samples]
        col_indices = torch.randperm(selected_layer.weight.shape[1])[:col_samples]
        weight = selected_layer.weight.data[row_indices][:, col_indices].detach().clone().to(device)
        
        print(f"Weight shape: {weight.shape}")
        print(f"Running scale search for {proj_type} with OSR={OSR}, first2={first2}")
        best_params = search_scale(weight, OSR=OSR, first2=first2)
        

        results[proj_type] = {
            "scale": best_params[0],
            "percentile_scale": best_params[1],
            "percentile_cut": best_params[2]
        }
    del model
    torch.cuda.empty_cache()
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find optimal scale parameters for model quantization")
    parser.add_argument("--model_path", type=str, default="/data/xjh/model_weight/opt/opt-6.7b", 
                   help="Path to the model (default: /data/xjh/model_weight/opt/opt-1.3b)")
    parser.add_argument("--OSR", type=int, default=2, help="Oversampling rate")
    parser.add_argument("--groupsize", type=int, default=128, help="groupsize")
    parser.add_argument("--n_sample", type=int, default=16, help="n_sample")
    parser.add_argument("--first2", action="store_true", help="Use first2 option")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    args = parser.parse_args()
    
    results = analyze_model_params(
        model_path=args.model_path,
        OSR=args.OSR,
        first2=args.first2,
        device=args.device,
        groupsize=args.groupsize,
        n_sample=args.n_sample
    )
    
    print("\nFinal Results:")
    for proj_type, params in results.items():
        print(f"{proj_type}: scale={params['scale']}, percentile_scale={params['percentile_scale']}, percentile_cut={params['percentile_cut']}")
    print("\nDone.")