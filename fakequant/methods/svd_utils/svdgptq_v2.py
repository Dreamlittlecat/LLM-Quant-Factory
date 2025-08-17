import transformers
import torch
import torch.nn as nn
import math
from sklearn.decomposition import NMF
import numpy as np
import sys
sys.path.append("./svd_utils")
from hadamard import random_hadamard_matrix
from kmeans_utils import RowWiseKMeansQuantizerTorch,KMeansQuantizerTorch

def svd(A,n=1):
    A_sign=torch.sign(A)
    A=torch.abs(A)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    A_approx=0
    for i in range(n):
        A_approx+=S[i] * torch.outer(U[:, i], Vh[i, :])
    return A_approx*A_sign

def svd_2(A, n=1, n_clusters=16, max_iter=3):
    A_sign = torch.sign(A)
    A = torch.abs(A)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)  

    combined_vectors = []
    for i in range(n):
        u = U[:, i]  # Row vector
        v = Vh[i, :]  # Column vector
        combined_vectors.append(torch.cat((u, v), dim=0))  # Concatenate u and v

    combined_vectors = torch.stack(combined_vectors, dim=0)  # Shape: (n, rows + cols)
    quantizer = KMeansQuantizerTorch(n_clusters=n_clusters, max_iter=max_iter)
    quantizer.fit(combined_vectors)
    quantized_vectors = quantizer.quantize(combined_vectors)

    for i in range(n):
        quantized_u = quantized_vectors[i, :U.shape[0]]  # Extract quantized u
        quantized_v = quantized_vectors[i, U.shape[0]:]  # Extract quantized v
        U[:, i] = quantized_u
        Vh[i, :] = quantized_v
       
    A_approx = 0
    for i in range(n):
        A_approx += S[i] * torch.outer(U[:, i], Vh[i, :])

    return A_approx * A_sign



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


class SVD_quant:
    def __init__(self, layer,disable_gptq):
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
        self.disable_gptq=disable_gptq
        pass


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
                    res_round=2,
                    rank=2,
                    n_clusters=16,
                    ):
        
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        H = self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1 
        W[:, dead] = 0
        del self.H
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        if blocksize == -1:
            blocksize= self.columns

        W_org = W.clone()
        #方块更新
        row_blocksize=blocksize
        #blocksize=256
                                                                                                                                                                                                                                                                                                         
        #Had=random_hadamard_matrix(blocksize,device=W.device).float()
        
        #按行分块
        for rowblocki, row_st in enumerate(range(0, self.rows, row_blocksize)):
            row_ed = min(row_st + row_blocksize, self.rows)
            W=W_org[row_st:row_ed,:]
       
            #按列块更新
            for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
                col_ed = min(col_st + blocksize, self.columns)
                n_cols = col_ed - col_st

                W1 = W[:, col_st:col_ed].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

                #W1=W1@Had
                for i in range(res_round):
                    res=W1-Q1
                    if n_clusters==-1:
                        res=svd(res,n=rank)
                    else:
                        res=svd_2(res,n=rank,n_clusters=n_clusters)
                    Q1+=res
                # Q1=Q1@Had.T
                # W1=W1@Had.T
    
                # for i in range(n_cols):
                #     w = W1[:, i]
                #     d = Hinv1[i, i]
                #     q=Q1[:,i]
                #     err1 = (w - q) / d
                #     Err1[:, i] = err1 
                # W[:, col_st:col_ed] = Q1
                W[:, col_st:col_ed] = Q1
                Err1=(W1-Q1)/torch.diag(Hinv1)
                if not self.disable_gptq:
                    W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])
            W_org[row_st:row_ed,:]=W
     
        # if isinstance(self.layer, transformers.Conv1D):
        #     W = W.t()
        self.layer.weight.data = W_org.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
 
        del W1, Q1, W, Err1, Hinv1,W_org
        del H, Hinv
        torch.cuda.empty_cache()
        return
    
    def free(self):
        self.H = None
        torch.cuda.empty_cache()