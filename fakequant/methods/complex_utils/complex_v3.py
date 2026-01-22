import transformers
import torch
import torch.nn as nn
import math
from sklearn.decomposition import NMF
import numpy as np
import sys
import math
from typing import Tuple
import torch
import math
import time


# sys.path.append("./complex_utils")
sys.path.append("/home/xjh/research/AI_xjh_research/llm_quant_factory/fakequant/methods/complex_utils")
# from arb_utils import high_order_residual_alternating_order2_rc_nomean,high_order_residual_alternating_order1_rc_nomean
from rtn_utils import pseudo_quantize_tensor
from kmeans_utils import RowWiseKMeansQuantizerTorch,KMeansQuantizerTorch


# from bi_utils import bi_mask_quant,high_order_residual
# from hadamard import random_hadamard_matrix

Flag_block_transform=False#调试用

def fast_blockwise_SVD_quant(matrix: torch.Tensor, n_bits_angle: int = 2,quantizer=None) -> torch.Tensor:
    """
    高效张量化版本：对矩阵每个 2x2 block 进行 SVD 量化。
    确保 U, V 为纯旋转矩阵，角度量化严格对应 2^n 个状态。
    """
    tick = time.time()
    rows, cols = matrix.shape
    block_size = 2
    if rows % block_size != 0 or cols % block_size != 0:
        raise ValueError("Matrix dimensions must be divisible by 2.")

    # 1. 重塑矩阵为 (N, 2, 2) 的批处理形式
    h_blocks, w_blocks = rows // block_size, cols // block_size
    blocks = matrix.view(h_blocks, block_size, w_blocks, block_size)
    blocks = blocks.permute(0, 2, 1, 3).reshape(-1, block_size, block_size)

    # 2. 批量进行 SVD 分解
    # 注意：Vh 是共轭转置，对于实数矩阵即为 V.T
    U, S, Vh = torch.linalg.svd(blocks)
    V = Vh.transpose(-2, -1)

    # 3. 修正行列式 (Determinant Correction)
    # 第一步：确保 det(U) == 1。若为 -1，翻转 U 和 V 的第二列，乘积 A 不变
    det_u = torch.linalg.det(U)
    flip_u = det_u < 0
    U[flip_u, :, 1] *= -1
    V[flip_u, :, 1] *= -1

    # 第二步：确保 det(V) == 1。若为 -1，翻转 V 的第二列和 S 的第二个值，乘积 A 不变
    det_v = torch.linalg.det(V)
    flip_v = det_v < 0
    V[flip_v, :, 1] *= -1
    S[flip_v, 1] *= -1  # 符号吸收：镜像效果转移到奇异值上

    # 4. 批量提取旋转角度 (弧度)
    # 此时 U, V 均为标准旋转矩阵 [[cos, -sin], [sin, cos]]
    theta_u = torch.atan2(U[:, 1, 0], U[:, 0, 0])
    theta_v = torch.atan2(V[:, 1, 0], V[:, 0, 0])

    # 5. 弧度空间直接量化 (确保 2^n 个状态)
    print("angle bits:", n_bits_angle)
    two_pi = 2 * math.pi
    num_levels = 2 ** n_bits_angle
    step = two_pi / num_levels

    # 将 (-pi, pi] 映射到 [0, 2pi)，然后计算量化索引，最后映射回弧度
    def quantize_theta(theta):
        # 映射到正数区间并计算索引
        indices = torch.round((theta % two_pi) / step) % num_levels
        return indices * step

    q_theta_u = quantize_theta(theta_u)
    q_theta_v = quantize_theta(theta_v)

    # 6. 构建量化后的旋转矩阵 U_q 和 V_q
    cos_u, sin_u = torch.cos(q_theta_u), torch.sin(q_theta_u)
    cos_v, sin_v = torch.cos(q_theta_v), torch.sin(q_theta_v)

    # 更加高效的构建方式：预分配空间再赋值
    U_q = torch.empty_like(U)
    U_q[:, 0, 0] = cos_u; U_q[:, 0, 1] = -sin_u
    U_q[:, 1, 0] = sin_u; U_q[:, 1, 1] = cos_u

    V_q = torch.empty_like(V)
    V_q[:, 0, 0] = cos_v; V_q[:, 0, 1] = -sin_v
    V_q[:, 1, 0] = sin_v; V_q[:, 1, 1] = cos_v

    time_tock = time.time()
    print(f"SVD 量化时间: {time_tock - tick:.4f} 秒")

    tick=time.time()
    # 7. 批量重构 block: A = U_q @ diag(S) @ V_q.T
    # 利用广播机制加速：(U_q 逐列乘以 S) @ V_q.T
    # S.unsqueeze(1) 形状为 (N, 1, 2)
    # print(S.shape)

    #整体进行量化
    # if quantizer is not None:
    #     S_flat = S.view(-1, 2)
    #     quantizer.fit(S_flat)
    #     S_quant_flat = quantizer.quantize(S_flat)
    #     S = S_quant_flat.view_as(S)
    # else:
    #     raise NotImplementedError("需要传入量化器对象")
    #     S = pseudo_quantize_tensor(S, num_bits=8, symmetric=True, per_channel=False)
    #先量化第一个元素，再量化第二个元素
    if quantizer is not None:
        # 记录原始形状以便最后还原
        orig_shape = S.shape
        # 将 S 转换为 (N, 2) 的形状
        S_flat = S.view(-1, 2)
        
        # 1. 提取第一个元素 s0 和第二个元素 s1
        s0 = S_flat[:, 0:1]
        s1 = S_flat[:, 1:2]

        # 2. 先量化第一个元素 s0

        quantizer.fit(s0)
        s0_quant = quantizer.quantize(s0)

        # 3. 计算比例：第二个元素 / 未量化前的第一个元素
        # 使用 1e-9 防止除以零
        eps = 1e-9
        ratio = s1 / (s0 + eps)

        # 4. 量化这个比例 ratio

        quantizer.fit(ratio)
        ratio_quant = quantizer.quantize(ratio)

        # 5. 根据量化后的比例和原始第一个元素还原第二个元素
        s1_recovered = ratio_quant * s0

        # 6. 合并结果并还原为初始形状
        S_quant_flat = torch.cat([s0_quant, s1_recovered], dim=1)
        S = S_quant_flat.view(orig_shape)
        
    else:
        # 保持你要求的 else 逻辑
        # 注：在 raise 之后的代码不会被执行，仅作为逻辑占位
        raise NotImplementedError("需要传入量化器对象")
        S = pseudo_quantize_tensor(S, num_bits=8, symmetric=True, per_channel=False)
    time_tock = time.time()
    print(f"SVD 奇异值量化时间: {time_tock - tick:.4f} 秒")
    
    res_blocks = (U_q * S.unsqueeze(1)) @ V_q.transpose(-2, -1)
    
    # 8. 恢复原始大矩阵形状
    quantized_matrix = res_blocks.view(h_blocks, w_blocks, block_size, block_size)
    quantized_matrix = quantized_matrix.permute(0, 2, 1, 3).reshape(rows, cols)
    # print(quantized_matrix.shape)
    # raise NotImplementedError("调试用，防止误运行")
    return quantized_matrix


import torch
import math
import time

def fast_n_by_n_block_svd_quant(matrix: torch.Tensor, n: int = 4, n_bits_param: int = 4, quantizer=None) -> torch.Tensor:
    """
    通用 n*n 块 SVD 量化方案
    :param matrix: 输入的大矩阵
    :param n: 块大小 (n*n)
    :param n_bits_param: U, V 矩阵参数的量化比特数
    :param quantizer: 外部传入的量化器对象 (需实现 fit 和 quantize)
    """

    tick = time.time()
    rows, cols = matrix.shape
    if rows % n != 0 or cols % n != 0:
        raise ValueError(f"Matrix dimensions must be divisible by {n}.")

    # 1. 分块处理 (Batching)
    h_blocks, w_blocks = rows // n, cols // n
    blocks = matrix.view(h_blocks, n, w_blocks, n)
    blocks = blocks.permute(0, 2, 1, 3).reshape(-1, n, n)
    num_blocks = blocks.shape[0]

    # 2. 批量 SVD
    U, S, Vh = torch.linalg.svd(blocks)
    V = Vh.transpose(-2, -1)

    # 3. 行列式修正 (Determinant Correction)
    # 强制使 U, V 属于 SO(n)，即 det=1，确保 Cayley 变换有效
    det_u = torch.linalg.det(U)
    U[det_u < 0, :, -1] *= -1
    V[det_u < 0, :, -1] *= -1

    det_v = torch.linalg.det(V)
    V[det_v < 0, :, -1] *= -1
    S[det_v < 0, -1] *= -1 # 符号吸收

    # 4. 使用 Cayley 变换参数化 U 和 V
    # Cayley 公式: Q = (I - A)(I + A)^-1  => A = (I - Q)(I + Q)^-1
    # A 是反对称矩阵 (A^T = -A)，只需要存储上三角 n(n-1)/2 个元素
    def matrix_to_params(Q):
        I = torch.eye(n, device=Q.device).unsqueeze(0)
        # 为防止 I+Q 不可逆（特征值为-1），加入微小扰动
        A = torch.linalg.solve(I + Q + 1e-6 * I, I - Q)
        # 提取严格上三角元素
        triu_indices = torch.triu_indices(n, n, offset=1)
        params = A[:, triu_indices[0], triu_indices[1]]
        return params

    def params_to_matrix(params):
        A = torch.zeros(num_blocks, n, n, device=params.device)
        triu_indices = torch.triu_indices(n, n, offset=1)
        A[:, triu_indices[0], triu_indices[1]] = params
        A = A - A.transpose(-2, -1) # 构造反对称矩阵
        I = torch.eye(n, device=params.device).unsqueeze(0)
        # Q = (I - A)(I + A)^-1
        Q = torch.linalg.solve(I + A, I - A)
        return Q

    # 提取 U, V 的参数
    params_u = matrix_to_params(U)
    params_v = matrix_to_params(V)
    # print(f"params_u shape: {params_u.shape}, params_v shape: {params_v.shape}")
    # raise Exception("Debug Stop")
    # 5. 参数量化 (类似角度量化)
    # 这里可以使用简单的均匀量化或传入的 quantizer
    def quantize_params(p, bits):
        scale = p.abs().max() + 1e-5
        levels = 2 ** bits
        step = 2 * scale / levels
        p_q = torch.round(p / step) * step
        return p_q

    q_params_u = quantize_params(params_u, n_bits_param)
    q_params_v = quantize_params(params_v, n_bits_param)

    # 重构量化后的 U 和 V
    U_q = params_to_matrix(q_params_u)
    V_q = params_to_matrix(q_params_v)
    # U_q=U
    # V_q=V

    # 6. 奇异值 S 的链式比例量化 (Chain-Ratio Quantization)
    if quantizer is not None:
        print("use quantizer")
        # S 形状为 (num_blocks, n)
        S_quant = torch.zeros_like(S)
        
        # 量化第一个奇异值 s0
        s0 = S[:, 0:1]
        quantizer.fit(s0)
        S_quant[:, 0:1] = quantizer.quantize(s0)
        
        # 链式量化后续比例: r_i = s_i / s_{i-1}
        last_s = s0
        for i in range(1, n):
            si = S[:, i:i+1]
            ratio = si / (last_s + 1e-9)
            quantizer.fit(ratio)
            r_quant = quantizer.quantize(ratio)
            S_quant[:, i:i+1] = r_quant * S_quant[:, i-1:i] # 基于前一个已量化的值还原
            last_s = si 
        S = S_quant
    else:
        # 降级方案：简单量化
        S = torch.clamp(S, min=1e-5) # 保证比例计算安全

    #7. 重构矩阵

    res_blocks = (U_q * S.unsqueeze(1)) @ V_q.transpose(-2, -1)

    # 8. 恢复形状
    quantized_matrix = res_blocks.view(h_blocks, w_blocks, n, n)
    quantized_matrix = quantized_matrix.permute(0, 2, 1, 3).reshape(rows, cols)
    
    print(f"n={n} SVD 量化完成，耗时: {time.time() - tick:.4f}s")
    return quantized_matrix






class complex_quant:
    def __init__(self, layer,cluster_m=4,cluster_p=16,group_size=-1,disable_complex=False):
        self.Clomplex_flag = not disable_complex
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        #group_size= 128*4
        #group_size=-1
        self.quantizer_m = RowWiseKMeansQuantizerTorch(n_clusters=cluster_m, group_size=group_size,max_iter=3)
        #group_size=128,针对—2-4
        self.quantizer_p = RowWiseKMeansQuantizerTorch(n_clusters=cluster_p, group_size=group_size,max_iter=5)

        #面向非复数域量化的kmeans量化器
        self.quantizer = RowWiseKMeansQuantizerTorch(n_clusters=2,group_size=128,max_iter=2)#调试用

        
        #per_bits= (math.log2(cluster_m)+math.log2(cluster_p))/2+(cluster_m+cluster_p)*16/2/group_size+(64+self.rows)*16/(64*self.rows*2)
        #print(f"blocksize:128,per weight bits:{per_bits},group_size:{group_size},cluster_1:{cluster_m},cluster_2:{cluster_p}")
        

    def add_batch(self, inp, out, blocksize=1024):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):  
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()  
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(self,
                    blocksize=-1, 
                    percdamp=0.01,
                    res_round=5,#调试用
                    ):
  
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        #处理hessian矩阵
   
        H = self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1 
        W[:, dead] = 0
        del self.H
       # Losses = torch.zeros(self.rows, device=self.dev)
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        if blocksize == -1:
            blocksize= self.columns
        print(f"使用blocksize: {blocksize}进行量化")
        #在blocksize不为-1时，采用gptq的补偿更新方法
        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            W1 = W[:, col_st:col_ed].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
          #  Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

            #进行V3版本的复数量化
            if self.Clomplex_flag:
                Q1=fast_blockwise_SVD_quant(W1,n_bits_angle=3,quantizer=self.quantizer)
                #Q1=fast_n_by_n_block_svd_quant(W1,n=4,n_bits_param=4,quantizer=self.quantizer)
            else:   
                #Had1=random_hadamard_matrix(W1.shape[-1],device=self.dev).float()
                #W1 = W1.matmul(Had1)
                self.quantizer.fit(W1.T)
                Q1 = self.quantizer.quantize(W1.T).T
                #Q1= Q1.matmul(Had1.t())

                1
            W[:, col_st:col_ed] = Q1
            Err1=(W1-Q1)/torch.diag(Hinv1)
            W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])#进行补偿更新
  

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        del W1, Q1, W, Err1, Hinv1
        #del Losses1
        del H, Hinv
        torch.cuda.empty_cache()
        return #{"error": torch.sum(Losses).item()}
    
    def free(self):
        self.H = None
        torch.cuda.empty_cache()
