import transformers
import torch
import torch.nn as nn
import math
from sklearn.decomposition import NMF
import numpy as np
import sys
sys.path.append("./complex_utils")
from arb_utils import high_order_residual_alternating_order2_rc_nomean,high_order_residual_alternating_order1_rc_nomean
from rtn_utils import pseudo_quantize_tensor
from kmeans_utils import RowWiseKMeansQuantizerTorch,KMeansQuantizerTorch
from bi_utils import bi_mask_quant,high_order_residual
from pb_utils import pb_mask
from hadamard import random_hadamard_matrix

#cluster_quantizer1=RowWiseKMeansQuantizerTorch(n_clusters=2, group_size=128,max_iter=3)
#cluster_quantizer2=RowWiseKMeansQuantizerTorch(n_clusters=8, group_size=128,max_iter=3)

#整个量化为一组，不分组
#cluster_quantizer1=KMeansQuantizerTorch(n_clusters=2, max_iter=10)
# cluster_quantizer2=KMeansQuantizerTorch(n_clusters=8, max_iter=10)
# cluster_quantizer=KMeansQuantizerTorch(n_clusters=4, max_iter=10)

def find_closest_factors(n):
    factors = []
    for i in range(1, int(n**0.5) + 1):  # 遍历从 1 到 sqrt(n)
        if n % i == 0:  # 如果 i 是 n 的因数
            factors.append((i, n // i))  # 添加因数对 (i, n // i)

    # 找到一个因数尽量是另一个的 2 倍的因数对
    closest_pair = None
    min_diff = float('inf')
    for f1, f2 in factors:
        if abs(f2 - 2 * f1) < min_diff:  # 比较差值，寻找最接近的因数对
            closest_pair = (f1, f2)
            min_diff = abs(f2 - 2 * f1)
    return closest_pair[-1]

def svd(A,n=1):
    A_sign=torch.sign(A)
    A=torch.abs(A)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    for i in range(n):
        A_approx=S[i] * torch.outer(U[:, i], Vh[i, :])
    #A_approx=S[0] * torch.outer(U[:, 0], Vh[0, :]) #+ S[1] * torch.outer(U[:, 1], Vh[1, :])+S[2] * torch.outer(U[:, 2], Vh[2, :])
    return A_approx*A_sign

def svd_kmeans(A,n=1):
    A_sign=torch.sign(A)
    A=torch.abs(A)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    u=U[:,0]
    v=Vh[0,:]
    quantizer=RowWiseKMeansQuantizerTorch(n_clusters=64, group_size=-1)
    quantizer.fit(u.unsqueeze(0))
    u=quantizer.quantize(u.unsqueeze(0)).squeeze()
    quantizer.fit(v.unsqueeze(0))
    v=quantizer.quantize(v.unsqueeze(0)).squeeze()
    A_approx=torch.outer(u,v)*S[0]
    return A_approx*A_sign

def svd_adakmeans(A,n=1):
    rows=A.shape[0]
    cols=A.shape[1]
    A_sign=torch.sign(A)
    A=torch.abs(A)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    u=U[:,0]
    v=Vh[0,:]
    k_cluster=int((2*(rows*cols*0.1)//(rows+cols)))
    if k_cluster<1:
        k_cluster=1
    k_cluster=2**k_cluster

    #print("k_cluster=",k_cluster)
    quantizer=KMeansQuantizerTorch(n_clusters=k_cluster,max_iter=3)
    u_v=torch.cat((u,v),-1)
    quantizer.fit(u_v.unsqueeze(0))
    u_v=quantizer.quantize(u_v.unsqueeze(0)).squeeze()
    u=u_v[:rows]
    v=u_v[rows:]
    A_approx=torch.outer(u,v)*S[0]
    return A_approx*A_sign

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
    def __init__(self, layer,cluster_1=2,cluster_2=8,group_size=-1,disable_complex=False):
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
   
        
        #self.quantizer_m=RowWiseKMeansQuantizerTorch(n_clusters=cluster_1, group_size=128,max_iter=3)#逐行量化，可以提高量化效果，但是速度会变慢
        self.quantizer_m=KMeansQuantizerTorch(n_clusters=cluster_1, max_iter=3)#整个张量量化
        self.quantizer_p=KMeansQuantizerTorch(n_clusters=cluster_2, max_iter=5)#整个张量量化

        # old version
        # self.quantizer1 = RowWiseKMeansQuantizerTorch(n_clusters=cluster_1, group_size=group_size)
        # self.quantizer2 = RowWiseKMeansQuantizerTorch(n_clusters=cluster_2, group_size=group_size)
        self.quantizer = KMeansQuantizerTorch(n_clusters=3, max_iter=3)#使用vector_kmeans方法时的量化器
        
    def vector_kmeans(self,w,p=32): #p=32是一个经验值，根据实际需要调整
        #p=32
        
        org_shape = w.size()
        w=w.reshape(-1,p)

        self.quantizer.fit(w)
        q=self.quantizer.quantize(w)
        q=q.reshape(org_shape)
        return q
    
       
    def svd_res_cluster(self,A,n=1,p=32): #p=32是一个经验值，根据实际需要调整
        org_shape=A.size()
        A=A.reshape(-1,p)
        # print("A.shape=",A.shape)
        #raise ValueError("stop")
        A_m,A_p=split_even_odd_columns_torch(A)

    #magnitude matrix,residual clustering
        #A_m_Q=nmf(A_m,n=n)
        #A_m_Q=svd(A_m,n=n)
        A_m_Q=svd_adakmeans(A_m,n=1)
        #A_m_Q=0
        res=A_m-A_m_Q

        #其他尝试，效果一般，不如聚类
        #res=bi_mask_quant(res)
        #res=high_order_residual_alternating_order1_rc_nomean(res,order=1,iter=15)#该方法由于res矩阵较小，开销太大
        #res=pseudo_quantize_tensor(res,n_bit=2,q_group_size=-1)

        #cluster quant
        self.quantizer_m.fit(res)
        res=self.quantizer_m.quantize(res)
        A_m_Q+=res

    #phase matrix: no residual，cluster quant
        #A_p_Q=nmf(A_p,n=n)
        #A_p_Q=svd_adakmeans(A_p,n=1)*0.05
        A_p_Q=0
        res=A_p-A_p_Q

        # print("res.shape=",res.shape)
        # torch.save(res,"./test_data/A_p.pt")
        # raise ValueError("stop")
        
        #其他尝试，效果一般，不如聚类
        #res=high_order_residual_alternating_order2_rc_nomean(res,order=2,iter=15)
        #res=res=pseudo_quantize_tensor(res,n_bit=3,q_group_size=-1)

        #cluster quant
        self.quantizer_p.fit(res)
        res=self.quantizer_p.quantize(res)
        A_p_Q+=res
 
        A_approx=reconstruct_weight_matrix_torch(A_m_Q,A_p_Q)
        return A_approx.reshape(org_shape)
        

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
                    res_round=5
                    ):
        
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        p=find_closest_factors(W.shape[0])
        #处理hessian矩阵
   
        H = self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1 
        W[:, dead] = 0
        del self.H
        Losses = torch.zeros(self.rows, device=self.dev)
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        if blocksize == -1:
            blocksize= self.columns
            
        #可不可以先加hadamard矩阵变换再用kmeans聚类量化?
        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            n_cols = col_ed - col_st
            W1 = W[:, col_st:col_ed].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]


            for i in range(n_cols):
                w = W1[:, i]
                d = Hinv1[i, i]
                #complex分解量化方法
                q=self.svd_res_cluster(w,n=1,p=p)
                #q=self.vector_kmeans(w,p=32)#聚类量化列向量
                Q1[:, i] = q
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1 

            W[:, col_st:col_ed] = Q1
            W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        del W1, Q1, W, Err1, Hinv1
        del H, Hinv
        torch.cuda.empty_cache()
        return 
    
    def free(self):
        self.H = None
        torch.cuda.empty_cache()
