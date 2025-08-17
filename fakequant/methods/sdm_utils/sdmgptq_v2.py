import transformers
import torch
import torch.nn as nn
import math
import numpy as np
import sys
sys.path.append("./sdm_utils")
from sdm import sdm_quant
from pb_utils import pb_mask
from hadamard import random_hadamard_matrix

class sdmgptq_quant:
    def __init__(self, layer,hadamard=True,pb_mask=False,metric="magnitude"):
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
        if hadamard:
            self.hadamard=random_hadamard_matrix(self.columns,device=self.dev).float()
            #self.hadamard=None
        else:
            self.hadamard=None
        self.pbmask=pb_mask
        assert hadamard != self.pbmask
        if metric not in ["magnitude","hessian"]:
            raise ValueError
        self.metric=metric

    def add_batch(self, inp, out, blocksize=1024):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]


        if self.hadamard is not None:
            inp = inp.float().matmul(self.hadamard.t())
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
                    OSR=3,
                    first2=False,
                    disable_gptq=False,
                    ):
        
        W = self.layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        if self.hadamard is not None:
            W = W.matmul(self.hadamard)
        #处理hessian矩阵
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
        #在blocksize不为-1时，采用gptq的补偿更新方法
        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            W1 = W[:, col_st:col_ed].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]   
            if self.pbmask:
                mask=pb_mask(W1,Hinv1,salient_metric=self.metric,low_frac=0.99)
                Q1=sdm_quant(W1*mask,OSR=OSR)+W1*(~mask)
                Q1+=(~mask)*W1
            else:
                Q1=sdm_quant(W1,OSR=OSR,first2=first2)

            W[:, col_st:col_ed] = Q1
            Err1=(W1-Q1)/torch.diag(Hinv1)
            if not disable_gptq:
                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])
  
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        if self.hadamard is not None:
            W = W.matmul(self.hadamard.t())
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