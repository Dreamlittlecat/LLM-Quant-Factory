import transformers
import torch
import torch.nn as nn
import math
from sklearn.decomposition import NMF
import numpy as np
import sys
import math
sys.path.append("./methods/complex_utils")
# from arb_utils import high_order_residual_alternating_order2_rc_nomean,high_order_residual_alternating_order1_rc_nomean
from rtn_utils import pseudo_quantize_tensor
from kmeans_utils import RowWiseKMeansQuantizerTorch,KMeansQuantizerTorch
# from bi_utils import bi_mask_quant,high_order_residual
# from hadamard import random_hadamard_matrix

Flag_block_transform=False#调试用

def svd_rtnquant(A):
    A_sign=torch.sign(A)
    A=torch.abs(A)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    u=U[:,0]
    v=Vh[0,:]
    u=pseudo_quantize_tensor(u.unsqueeze(0),16,128).squeeze()
    v=pseudo_quantize_tensor(v.unsqueeze(0),16,128).squeeze()
    A_approx=torch.outer(u,v)*S[0]
    #A_approx=S[0] * torch.outer(U[:, 0], Vh[0, :]) #+ S[1] * torch.outer(U[:, 1], Vh[1, :])+S[2] * torch.outer(U[:, 2], Vh[2, :])
    return A_approx*A_sign

def svd_kmeans(A,n=1):
    A_sign=torch.sign(A)
    A=torch.abs(A)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    u=U[:,0]
    v=Vh[0,:]
    quantizer=RowWiseKMeansQuantizerTorch(n_clusters=64, block_size=-1)
    quantizer.fit(u.unsqueeze(0))
    u=quantizer.quantize(u.unsqueeze(0)).squeeze()
    quantizer.fit(v.unsqueeze(0))
    v=quantizer.quantize(v.unsqueeze(0)).squeeze()
    A_approx=torch.outer(u,v)*S[0]
    return A_approx*A_sign

def svd_kmeans_v2(A,n=1):
    A_sign=torch.sign(A)
    A=torch.abs(A)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    quantizer=RowWiseKMeansQuantizerTorch(n_clusters=64, block_size=-1)
    for i in range(n):
        u=U[:,i]
        v=Vh[i,:]
        quantizer.fit(u.unsqueeze(0))
        u=quantizer.quantize(u.unsqueeze(0)).squeeze()
        quantizer.fit(v.unsqueeze(0))
        v=quantizer.quantize(v.unsqueeze(0)).squeeze()
        A_approx=S[i] * torch.outer(u,v)
    return A_approx*A_sign

def svd(A,n=1):
    A_sign=torch.sign(A)
    A=torch.abs(A)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    for i in range(n):
        A_approx=S[i] * torch.outer(U[:, i], Vh[i, :])
    #A_approx=S[0] * torch.outer(U[:, 0], Vh[0, :]) #+ S[1] * torch.outer(U[:, 1], Vh[1, :])+S[2] * torch.outer(U[:, 2], Vh[2, :])
    return A_approx*A_sign

#nmf的效果略好于svd
def nmf(A,n=1,debug=False):
    nmf_model = NMF(n_components=n, init='random', random_state=0)
    device = A.device
    data_type = A.dtype
    A = A.cpu().detach().numpy()
    A_sign = np.sign(A)
    W = nmf_model.fit_transform(np.abs(A))
    H = nmf_model.components_
    if not debug:
        A_reconstructed = np.dot(W, H)*A_sign
        A_reconstructed = torch.tensor(A_reconstructed, device=device, dtype=data_type)
        return A_reconstructed
    else:
        W=torch.tensor(W, device=device, dtype=data_type)
        H=torch.tensor(H, device=device, dtype=data_type)
        A_sign=torch.tensor(A_sign, device=device, dtype=data_type)
        return W,H,A_sign

def split_even_odd_columns_torch(weight_matrix):
    # 选取奇数列和偶数列
    odd_columns = weight_matrix[:, ::2]  # 奇数列 (索引从 0 开始, 0,2,4,...)
    even_columns = weight_matrix[:, 1::2]  # 数列 (1,3,5,...)
    # 构造复数矩阵
    complex_matrix = odd_columns + 1j * even_columns
    # 计算幅值和相位
    magnitude_matrix = torch.abs(complex_matrix)  # 幅值
    phase_matrix = torch.angle(complex_matrix)  # 相位（以弧度表示）
    # 将相位调整到 [0, 2π) 范围
    phase_matrix = (phase_matrix + 2 * torch.pi) % (2 * torch.pi)
    return magnitude_matrix, phase_matrix

def reconstruct_weight_matrix_torch(magnitude_matrix, phase_matrix):
    # 构造复数矩阵
    complex_matrix = magnitude_matrix * torch.exp(1j * phase_matrix)
    # 提取奇数列和偶数列
    odd_columns = torch.real(complex_matrix)  # 实部作为奇数列
    even_columns = torch.imag(complex_matrix)  # 虚部作为偶数列
    # 合并奇数列和偶数列为原始权重矩阵
    rows, cols = odd_columns.shape
    reconstructed_matrix = torch.zeros((rows, cols * 2), dtype=odd_columns.dtype, device=odd_columns.device)
    reconstructed_matrix[:, ::2] = odd_columns  # 偶数列
    reconstructed_matrix[:, 1::2] = even_columns  # 奇数列
    return reconstructed_matrix


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

    
        #对复数权重的角度和幅值分别量化
        #group_size=-1
        self.quantizer_m = RowWiseKMeansQuantizerTorch(n_clusters=cluster_m, group_size=group_size,max_iter=3)
        #group_size=128,针对—2-4
        self.quantizer_p = RowWiseKMeansQuantizerTorch(n_clusters=cluster_p, group_size=group_size,max_iter=5)
        #对照组，直接对权重矩阵量化
        self.quantizer = RowWiseKMeansQuantizerTorch(n_clusters=4, group_size=group_size,max_iter=5)#调试用

        if group_size == -1:
            group_size = self.rows
        elif group_size == -2:
            group_size = self.rows//2
        elif group_size == -4:
            group_size = self.rows//4
        else:
            group_size = group_size
        
        #取blocksize=128时的每权重量化位数
        per_bits= (math.log2(cluster_m)+math.log2(cluster_p))/2+(cluster_m+cluster_p)*16/2/group_size+(64+self.rows)*16/(64*self.rows*2)
        print(f"blocksize:128,per weight bits:{per_bits},group_size:{group_size},cluster_1:{cluster_m},cluster_2:{cluster_p}")
        

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
                    blocksize=128, 
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
            
        #在blocksize不为-1时，采用gptq的补偿更新方法
        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            W1 = W[:, col_st:col_ed].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
          #  Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

            #是否进行复数量化
            if self.Clomplex_flag:
                #分解复数权重矩阵,complex量化
                if Flag_block_transform:
                    M1, P1 = split_even_odd_columns_torch(W1.T)
                else:
                    M1, P1 = split_even_odd_columns_torch(W1)

        #幅值量化
            #svd残差聚类量化幅值矩阵,nmf效果略好于svd
                M_Q1=svd(M1,n=1)
                #M_Q1=0
                #M_Q1=nmf(M1,n=1)
                res=M1-M_Q1
                self.quantizer_m.fit(res.T)
                res = self.quantizer_m.quantize(res.T).T#按列聚类
                #M_res_Q1 = pseudo_quantize_tensor(res, n_bit=4, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False)
                M_Q1+=res

            #nmf残差bi_mask量化幅值矩阵
                # M_Q1=nmf(M1,n=1)
                # res=M1-M_Q1
                # res=bi_mask_quant(res)
                # M_Q1+=res

            #rc残差聚类量化幅值矩阵
                # M_Q1=high_order_residual_alternating_order1_rc_nomean(M1,order=1,iter=3)
                # res=M1-M_Q1
                # self.quantizer1.fit(res)
                # res = self.quantizer1.quantize(res)
                # M_Q1+=res

            #svd残差rc量化幅值矩阵
                # M_Q1=svd(M1,n=1)
                # res=M1-M_Q1
                # res=high_order_residual_alternating_order2_rc_nomean(res,order=2,iter=15)
                # M_Q1+=res
        

            #rc残差rc量化幅值矩阵,效果一般
                # M_Q1=high_order_residual_alternating_order1_rc_nomean(M1,order=1,iter=3)
                # res=M1-M_Q1
                # res=high_order_residual_alternating_order2_rc_nomean(res,order=2,iter=3)
                # M_Q1+=res

            #递归残差svd量化幅值矩阵
                # M_Q1=torch.zeros_like(M1)
                # for i in  range(res_round):
                #     #print("i",i)
                #     res1=M1-M_Q1
                #     res_group1=torch.zeros_like(res1)
                #     mask1=res1>=0
                #     res_group1[mask1]=res1[mask1]
                #     #mean_value=torch.abs(torch.mean(res1))

                #     #res_group1+=mean_value
                #     res_group2=torch.zeros_like(res1)
                #     mask2=res1<0
                #     res_group2[mask2]=res1[mask2]
                #     #res_group2-=mean_value

                #     torch.save(torch.sign(res1),f"./M_sign_{i}.pt")
                #     # res_group1=svd(res_group1,n=1)
                #     # res_group2=svd(res_group2,n=1)
                #     # print((torch.all(res_group1>0)))
                #     # print((torch.all(res_group2<0)))
                #     res_group1=svd_kmeans(res_group1)#-mean_value
                #     res_group2=svd_kmeans(res_group2)#+mean_value
                #     M_Q1+=(res_group1+res_group2)
                    
                #raise ValueError


        #相位量化
            #聚类量化相位矩阵
                P_Q1=0
                res=P1-P_Q1
                self.quantizer_p.fit(res.T)
                res= self.quantizer_p.quantize(res.T).T
                P_Q1+=res




            #NMF残差bi_mask量化相位矩阵
                # P_Q1=nmf(P1,n=1)
                # res=P1-P_Q1
                # #res=bi_mask_quant(res)
                # res=high_order_residual(res)
                # P_Q1+=res


            #rc残差聚类量化相位矩阵
                # P_Q1=high_order_residual_alternating_order1_rc_nomean(P1,order=1,iter=3)
                # res=P1-P_Q1
                # self.quantizer2.fit(res)
                # res= self.quantizer2.quantize(res)
                # P_Q1+=res


            #svd残差rc量化相位矩阵
                # P_Q1=svd(P1,n=1)
                # res=P1-P_Q1
                # res=high_order_residual_alternating_order2_rc_nomean(res,order=2,iter=15)
                # P_Q1+=res

            #rc残差rc量化相位矩阵，效果一般
                # P_Q1=high_order_residual_alternating_order1_rc_nomean(P1,order=1,iter=3)
                # res=P1-P_Q1
                # res=high_order_residual_alternating_order2_rc_nomean(res,order=2,iter=3)
                # P_Q1+=res

            #尝试量化相位矩阵的余弦值,效果不好，不可用
                # P1_cosine = torch.cos(P1)
                # self.quantizer2.fit(P1_cosine)
                # P1_cosine= self.quantizer2.quantize(P1_cosine)
                # #P1_cosine = svd(P1_cosine,n=1)
                # P_Q1=torch.acos(P1_cosine)

            #递归残差svd量化相位矩阵,至少要10轮才能可用，代价太大，不可用
                # P_Q1=torch.zeros_like(P1)
                # for i in  range(res_round):
                #     res2=P1-P_Q1
                #     #res_mean=torch.abs(torch.mean(res2))
                #     res_group1=torch.zeros_like(res2)
                #     mask1=res2>=0
                #     res_group1[mask1]=res2[mask1]
                #     #res_group1+=res_mean
                #     res_group2=torch.zeros_like(res2)
                #     mask2=res2<0
                #     res_group2[mask2]=res2[mask2]

                #     torch.save(torch.sign(res2),f"./P_sign_{i}.pt")
                #     #res_group2-=res_mean
                #     #res_group2=(res_group2+2*torch.pi)%(2*torch.pi)
                #     # res_group1=svd(res_group1,n=1)
                #     # res_group2=svd(res_group2,n=1)


                #     res_group1=svd_kmeans(res_group1)#-res_mean
                #     res_group2=svd_kmeans(res_group2)#+res_mean
                #     P_Q1+=(res_group1+res_group2)

                #raise ValueError

                Q1 = reconstruct_weight_matrix_torch(M_Q1,P_Q1)
                if Flag_block_transform:
                    Q1=Q1.T
            else:   
                #Had1=random_hadamard_matrix(W1.shape[-1],device=self.dev).float()
                #W1 = W1.matmul(Had1)
                self.quantizer.fit(W1.T)
                Q1 = self.quantizer.quantize(W1.T).T
                #Q1= Q1.matmul(Had1.t())
                
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
