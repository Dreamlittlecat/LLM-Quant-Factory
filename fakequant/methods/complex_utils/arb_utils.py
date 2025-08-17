import torch

@torch.no_grad()
def high_order_residual_alternating_order2_rc_nomean(x, mask=None, order=2, iter=3): 
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask

    binary_list = []
    alpha_list_r = []
    alpha_list_c = []
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        # alpha row
        scale_tensor_all_r = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all_r = torch.where(torch.isnan(scale_tensor_all_r), torch.zeros_like(scale_tensor_all_r), scale_tensor_all_r)
        alpha_list_r.append(scale_tensor_all_r.clone())
        # alpha column
        scale_tensor_all_c = torch.nanmean(torch.abs(masked_x_tensor / scale_tensor_all_r[:, None]), dim=0)
        scale_tensor_all_c = torch.where(torch.isnan(scale_tensor_all_c), torch.zeros_like(scale_tensor_all_c), scale_tensor_all_c)
        alpha_list_c.append(scale_tensor_all_c.clone())

        binary= torch.sign(masked_x_tensor)
        binary_list.append(binary.clone())
        binary *= scale_tensor_all_r[:, None]
        binary *= scale_tensor_all_c[None, :]
        sum_order = sum_order + binary*mask

    # Alternating update
    sum_order_alternating = sum_order.clone()

    for k in range(iter):        
        # 2-1. Fix mean, alpha column, and B, update alpha row 0
        W_tilde = new_matrix - (alpha_list_c[1][None, :] * alpha_list_r[1][:, None] * binary_list[1]) * mask
        alpha_c_B = alpha_list_c[0][None, :] * binary_list[0] * mask
        alpha_list_r[0] = torch.sum(alpha_c_B * W_tilde, dim=1) / (torch.sum(alpha_c_B * alpha_c_B, dim=1) + 1e-8)
        
        # 2-2. Fix mean, alpha row, and B, update alpha column 0
        alpha_r_B =  alpha_list_r[0][:, None] * binary_list[0] * mask
        alpha_list_c[0] = torch.sum(alpha_r_B * W_tilde, dim=0) / (torch.sum(alpha_r_B * alpha_r_B, dim=0) + 1e-8)

        # 2-3. Fix mean, alpha column, and B, update alpha row 1
        W_tilde = new_matrix - (alpha_list_c[0][None, :] * alpha_list_r[0][:, None] * binary_list[0]) * mask
        alpha_c_B = alpha_list_c[1][None, :] * binary_list[1] * mask
        alpha_list_r[1] = torch.sum(alpha_c_B * W_tilde, dim=1) / (torch.sum(alpha_c_B * alpha_c_B, dim=1) + 1e-8)
        
        # 2-4. Fix mean, alpha row, and B, update alpha column 1
        alpha_r_B =  alpha_list_r[1][:, None] * binary_list[1] * mask
        alpha_list_c[1] = torch.sum(alpha_r_B * W_tilde, dim=0) / (torch.sum(alpha_r_B * alpha_r_B, dim=0) + 1e-8)

        # 3. Fix mean and alpha, update B
        new_matrix_expanded = new_matrix.unsqueeze(-1)
        comb0 = alpha_list_r[0].reshape(-1, 1) @ alpha_list_c[0].reshape(1, -1)
        comb1 = alpha_list_r[1].reshape(-1, 1) @ alpha_list_c[1].reshape(1, -1)
        v = torch.stack([-comb0 - comb1, -comb0 + comb1, 
                    comb0 - comb1, comb0 + comb1], dim=2)

        min_indices = torch.argmin(torch.abs(new_matrix_expanded - v), dim=-1)

        binary_list[0] = torch.ones_like(min_indices)
        binary_list[0][(min_indices == 0) | (min_indices == 1)] = -1
        binary_list[1] = torch.ones_like(min_indices)
        binary_list[1][(min_indices == 0) | (min_indices == 2)] = -1 

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + (alpha_list_c[0][None, :] * alpha_list_r[0][:, None] * binary_list[0] + alpha_list_c[1][None, :] * alpha_list_r[1][:, None] * binary_list[1]) * mask
    return sum_order_alternating



@torch.no_grad()
def high_order_residual_alternating_order1_rc_nomean(x, mask=None, order=1, iter=3):
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)
    # global index
    # index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        # alpha row
        scale_tensor_all_r = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all_r = torch.where(torch.isnan(scale_tensor_all_r), torch.zeros_like(scale_tensor_all_r), scale_tensor_all_r)
        # alpha column
        scale_tensor_all_c = torch.nanmean(torch.abs(masked_x_tensor / scale_tensor_all_r[:, None]), dim=0)
        scale_tensor_all_c = torch.where(torch.isnan(scale_tensor_all_c), torch.zeros_like(scale_tensor_all_c), scale_tensor_all_c)

        binary= torch.sign(masked_x_tensor)
        new_binary = binary.clone()
        binary *= scale_tensor_all_r[:, None]
        binary *= scale_tensor_all_c[None, :]  
        sum_order = sum_order + binary*mask

    # Alternating update
    sum_order_alternating = sum_order.clone()
    new_alpha_r = scale_tensor_all_r.clone()
    new_alpha_c = scale_tensor_all_c.clone()
    for k in range(iter):        
        # 1-1. Fix mean, alpha column, and B, update alpha row
        alpha_c_B = new_alpha_c[None, :] * new_binary * mask
        new_alpha_r = torch.sum(alpha_c_B * new_matrix, dim=1) / (torch.sum(alpha_c_B * alpha_c_B, dim=1) + 1e-8)
        
        # 1-2. Fix mean, alpha row, and B, update alpha column
        alpha_r_B = new_alpha_r[:, None] * new_binary * mask
        new_alpha_c = torch.sum(alpha_r_B * new_matrix, dim=0) / (torch.sum(alpha_r_B * alpha_r_B, dim=0) + 1e-8)

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + new_alpha_c[None, :] * new_alpha_r[:, None] * new_binary * mask

    return sum_order_alternating