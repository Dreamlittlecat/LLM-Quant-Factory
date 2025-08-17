import torch

def pb_mask(weight,H=None,salient_metric="hessian",low_frac=0.999):
    mask = torch.zeros_like(weight, dtype=torch.bool)
    if salient_metric == "hessian":
        tmp=(weight**2/(torch.diag(H).reshape(1,-1))**2)
        thresh=torch.sort(tmp.flatten())[0][int(tmp.numel()*low_frac)]
        mask=tmp<=thresh
        return mask
    elif salient_metric == "magnitude":
        saliency = torch.abs(weight)
        thresh = torch.sort(saliency.flatten())[0][int(saliency.numel() * low_frac)]
        mask = saliency <= thresh
        return mask
    else:
        raise NotImplementedError("salient_metric not implemented")

# def pb_mask(weight, H=None, salient_metric="hessian", low_frac=0.95):
#     mask = torch.zeros_like(weight, dtype=torch.bool)

#     if salient_metric == "hessian":
#         # 计算每列的显著性度量
#         tmp = (weight**2 / (torch.diag(H).reshape(1, -1))**2)
#         col_mean = tmp.mean(dim=0)  # 按列计算平均值
#         thresh = torch.sort(col_mean)[0][int(col_mean.numel() * low_frac)]  # 按列平均值计算阈值
#         col_mask = col_mean <= thresh  # 按列生成掩码

#         # 将列掩码扩展为与 weight 相同的形状
#         mask = col_mask.unsqueeze(0).expand_as(weight)
#         return mask

#     elif salient_metric == "magnitude":
#         # 计算每列的显著性度量
#         col_mean = torch.abs(weight).mean(dim=0)  # 按列计算平均值
#         thresh = torch.sort(col_mean)[0][int(col_mean.numel() * low_frac)]  # 按列平均值计算阈值
#         col_mask = col_mean <= thresh  # 按列生成掩码

#         # 将列掩码扩展为与 weight 相同的形状
#         mask = col_mask.unsqueeze(0).expand_as(weight)
#         return mask

#     else:
#         raise NotImplementedError("salient_metric not implemented")