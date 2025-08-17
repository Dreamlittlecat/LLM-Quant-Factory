import transformers
import torch
import torch.nn as nn
import math
from sklearn.decomposition import NMF
import numpy as np
import sys
sys.path.append("./svd")
from hadamard import random_hadamard_matrix


def svd(A,n=1):
    #return A
    A_sign=torch.sign(A)
    A=torch.abs(A)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    #print(S)
    A_approx=S[0] * torch.outer(U[:, 0], Vh[0, :]) #+ S[1] * torch.outer(U[:, 1], Vh[1, :])+S[2] * torch.outer(U[:, 2], Vh[2, :])
    return A_approx*A_sign

def nmf_svd(A,n=1,debug=True):
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



class SVD_quant:
    def __init__(self, layer):
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
                    res_round=20
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

            
        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            n_cols = col_ed - col_st

            W1 = W[:, col_st:col_ed].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

            #svd
            # for i in range(res_round):
            #     res=W1-Q1
            #     res=svd(res,n=1)
            #     Q1+=res

            #nmf
            # res_sign=torch.zeros_like(W1)
            # for i in range(res_round):
            #     res=W1-Q1
            #     res_w,res_h,temp_sign=nmf_svd(res)
            #     # if i<=1:
            #     res_sign=temp_sign
            #     #print(temp_sign)
            #     #res_sign=torch.sign(res_sign)
                
            #     res=res_w@res_h*res_sign#torch.dot(res_w,res_h)
            #     Q1+=res            

            #mask_nmf
            Had=random_hadamard_matrix(col_ed-col_st,device=W1.device).float()
            W1=W1@Had
            for i in range(res_round):
                res=W1-Q1
                res_group1=torch.zeros_like(res)
                res_group2=torch.zeros_like(res)
                res_group1[res>0]=res[res>0]
                res_group2[res<0]=res[res<0]
                # res_group1=nmf_svd(res_group1,n=1,debug=False)
                # res_group2=nmf_svd(res_group2,n=1,debug=False)
                res_group1=svd(res_group1,n=1)
                res_group2=svd(res_group2,n=1)
                res=res_group1+res_group2
                Q1+=res
            Q1=Q1@Had.T
            W1=W1@Had.T


   
            for i in range(n_cols):
                # shape of w: [oc, 1]
                w = W1[:, i]
                d = Hinv1[i, i]
                q=Q1[:,i]
                #q = torch.zeros_like(w)
                #Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2
                # breakpoint()
                err1 = (w - q) / d
                Err1[:, i] = err1 
            

            W[:, col_st:col_ed] = Q1
            Losses += torch.sum(Losses1, 1) / 2
            W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])
  

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
 
        del W1, Q1, W, Err1, Losses1, Hinv1
        del H, Hinv
        torch.cuda.empty_cache()
        return {"error": torch.sum(Losses).item()}
