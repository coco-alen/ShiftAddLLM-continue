import math
import time
import os

from tqdm import tqdm
import torch
import torch.nn as nn
import transformers

from quantizers.quant import *
from quantizers.bcq_quant.quantizer import quantize as bcq_quantize

from .quip_method import QuantMethod

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ(QuantMethod):

    def fasterquant(
        self, args, model_name, layer_name
    ):  
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        groupsize = args.groupsize
        tick = time.time()

        if args.static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if args.act_order:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        H = self.H
        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        for i1 in tqdm(range(0, self.columns, args.blocksize), desc=layer_name, leave=False):
            i2 = min(i1 + args.blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if args.lut_eval:
                    if groupsize != -1:
                        idx = i1 + i
                        if args.act_order:
                            idx = perm[idx]
                        group = idx // groupsize
                    else:
                        group = 0
                    alpha = self.quantizer.alpha[:,group,:].unsqueeze(1)
                    q, BinaryWeight = bcq_quantize(w.unsqueeze(1), alpha, groupsize=-1)
                    q = q.flatten()
                else:
                    if groupsize != -1:
                        if not args.static_groups:
                            if (i1 + i) % groupsize == 0:
                                self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                        else:
                            idx = i1 + i
                            if args.act_order:
                                idx = perm[idx]
                            self.quantizer = groups[idx // groupsize]

                    q = quantize(
                        w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if args.act_order:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        self.postproc()
        self.error_compute(W, self.layer.weight.data)
        
    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
