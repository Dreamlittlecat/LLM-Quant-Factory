from re import L
import numpy as np
from pyparsing import line
import torch

import torch

@torch.no_grad()
def high_order_residual(x, mask=None, order=2):
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask

    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)

        binary= torch.sign(masked_x_tensor)
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask
    
    return sum_order


def generate_structural_mask(origin_matrix, mask3, braq1_border):
    mask1_2 = ~mask3

    binary_group = torch.abs(origin_matrix*mask1_2)

    mask2 = binary_group >= braq1_border
    mask1 = binary_group < braq1_border

    mask1 = mask1 * mask1_2
    mask2 = mask2 * mask1_2

    return mask1, mask2

# def generate_mask(origin_matrix, braq2_border, braq1_border):
#     mask3 = torch.abs(origin_matrix) >= braq2_border
#     mask1 = torch.abs(origin_matrix) <= braq1_border
#     mask2 = (torch.abs(origin_matrix) > braq1_border) & (torch.abs(origin_matrix) < braq2_border)
#     return mask1, mask2, mask3

def error_computing(origin_matrix, quantized_matrix):
    mse = torch.mean((origin_matrix - quantized_matrix) ** 2)
    return mse

# def calculate_percentage_and_variance_original(weights, abs_weights, bin_edges):
#     percentages = []
#     variances = []
#     accum_percentages = [0]
#     total_elements = abs_weights.numel()
#     for i in range(len(bin_edges) - 1):
#         bin_mask = (abs_weights >= bin_edges[i]) & (abs_weights < bin_edges[i + 1])
#         bin_weights = weights[bin_mask]
#         percentages.append(bin_weights.numel() / total_elements * 100)
#         accum_percentages.append(accum_percentages[-1] + percentages[-1])
#         variances.append(torch.var(bin_weights))
#     return percentages, variances, accum_percentages


def structural_searching(origin_matrix, up_lim=30):
    minimal_value = float('inf')
    minimal_value_0 = float('inf')

    true_counts = origin_matrix.abs().sum(dim=0)

    error = []
    lines = []
    # search for the optimal split for the first group, high order=2,, structured search
    _, top_braq_2_columns = torch.topk(true_counts, up_lim)
    for i in range(1, up_lim):
        mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
        mask3[:, top_braq_2_columns[:i]] = True
        group3 = high_order_residual(origin_matrix, mask3, order=2)
        group4 = high_order_residual(origin_matrix, ~mask3, order=2)
        quantize_error_0 = error_computing(origin_matrix, group4+group3)
        error.append(quantize_error_0.item())
        lines.append(i)
        if quantize_error_0 < minimal_value_0:
            minimal_value_0 = quantize_error_0
            optimal_split_0 = i
    _, top_braq_2_columns = torch.topk(true_counts, optimal_split_0)
    # print(f"optimal split for the first group: {optimal_split_0}")
    # print(f"top_braq_2_columns: {top_braq_2_columns}")
    
    mask3 = torch.full((origin_matrix.shape[0], origin_matrix.shape[1]), False).to(origin_matrix.device)
    mask3[:, top_braq_2_columns] = True
    group3 = high_order_residual(origin_matrix, mask3, order=2)

    search_matrix = origin_matrix * (~mask3)

    flat_abs_tensor = torch.abs(search_matrix).view(-1)
    percentiles = torch.linspace(0.10, 0.90, 81).to(origin_matrix.device)
    percentile_values = torch.tensor(
        np.quantile(flat_abs_tensor.detach().cpu().numpy(), q=percentiles.cpu().numpy(), axis=None, keepdims=False)
    ).to(origin_matrix.device)

    # search for the optimal split for the second group, high order=1,, non-structured search
    for split_value in percentile_values:
        mask1, mask2 = generate_structural_mask(origin_matrix, mask3, split_value)
        group1 = high_order_residual(origin_matrix, mask1, order=1)
        group2 = high_order_residual(origin_matrix, mask2, order=1)

        quantize_error = error_computing(origin_matrix, group1+group2+group3)
        if quantize_error < minimal_value:
            minimal_value = quantize_error
            optimal_split = split_value
        tmp = torch.max(torch.abs(search_matrix)).item()
    
    return optimal_split, mask3

# def find_optimal_split(group_max, origin_matrix, border):
#     optimal_split = None
#     minimal_value = float('inf')
#     searching_steps = torch.arange(0.1,0.8,0.01)
#     searching_steps = searching_steps * group_max

#     group3 = high_order_residual(origin_matrix, torch.abs(origin_matrix) > border, order=2)
#     for split_value in searching_steps:

#         group1 = high_order_residual(origin_matrix, (torch.abs(origin_matrix) > split_value) & (torch.abs(origin_matrix) <= border), order=1)
#         group2 = high_order_residual(origin_matrix, torch.abs(origin_matrix) <= split_value, order=1)

#         quantize_error = error_computing(origin_matrix, group1+group2+group3)
#         if quantize_error < minimal_value:
#             minimal_value = quantize_error
#             optimal_split = split_value

#     return optimal_split, minimal_value


def structural_guassian_distribution(tmp, H=None, metric="magnitude", up_lim=10):
    if metric == "hessian":
        raise NotImplementedError
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    optimal_split, mask3 = structural_searching(target_weights, up_lim)
    mask1, mask2 = generate_structural_mask(target_weights, mask3, optimal_split)
    #print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    return mask1, mask2, mask3

def bi_mask_quant(weight):
    mask = torch.zeros_like(weight, dtype=torch.bool).unsqueeze(0).repeat_interleave(3, dim=0)
    mask1, mask2, mask3 = structural_guassian_distribution(weight)
    mask[0] = mask1
    mask[1] = mask2
    mask[2] = mask3
    q_part_groups = []
    orders=(1,1,2)
    for i in range(mask.shape[0]):
        q_part_groups.append(high_order_residual(weight, mask[i], order=orders[i]))
    Q=torch.zeros_like(weight)
    for j in range(mask.shape[0]):
        Q += q_part_groups[j][:] * mask[j, :]
    return Q
    pass