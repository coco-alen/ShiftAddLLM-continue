import numpy as np
import math
import torch
import torch.nn as nn

#  ================== int quant ==================

def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

def quant_int(x, bits, per_block=False):
    x = x.float()
    if per_block:
        seq_len, dim = x.shape
        assert dim % 8 == 0, "dim must be multiple of 8"
        x = x.view(seq_len , dim // 8, 8)

    maxq = torch.tensor(2 ** bits - 1, device=x.device, dtype=x.dtype)
    tmp = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
    xmin = torch.minimum(x.min(-1)[0], tmp)
    xmax = torch.maximum(x.max(-1)[0], tmp)
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1
    scale = (xmax - xmin) / maxq
    zero = torch.round(-xmin / scale)
    scale = scale.unsqueeze(-1)
    zero = zero.unsqueeze(-1)
    out = quantize(x, scale, zero, maxq)
    if per_block:
        out = out.reshape(seq_len, dim)
    out = out.to(torch.float16)
    return out

#  ================== fp quant ==================
def get_log_scale(x ,mantissa_bit, exponent_bit, interval=None):
    
    maxval = (2 - 2 ** (-mantissa_bit)) * 2 ** (
                2**exponent_bit - 1 - interval
            )
    bias = interval if interval != None else 2 ** (exponent_bit - 1)
    bias = bias.float()
    minval = -maxval
    a = torch.min(torch.max(x, minval), maxval)
    log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(a)) + bias)).detach(), 1.0)
    return a, 2.0 ** (log_scales - mantissa_bit - bias)
    


def quant_fp(x, bits, exponent_bit, per_block=False):
    mantissa_bit = bits - exponent_bit - 1
    assert mantissa_bit > 0, "mantissa_bit must be positive"
    x_maxval = x.abs().max()
    interval = 2**exponent_bit - torch.log2(x_maxval) + math.log2(2 - 2 ** (-mantissa_bit)) - 1

    if per_block:
        seq_len, dim = x.shape
        assert dim % 8 == 0, "dim must be multiple of 8"
        x = x.reshape(seq_len * dim // 8, 8)

    x, x_scale = get_log_scale(x, mantissa_bit, exponent_bit, interval)
    x_sim=(x/x_scale).round_()
    out = x_sim.mul(x_scale)

    if per_block:
        out = out.reshape(seq_len, dim)

    return out

#  ================== Structure ==================

class LinearWithActQuant(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 act_quant_int = 0, 
                 act_quant_fp = 0, 
                 act_quant_fp_exponent = 0,
                 act_quant_per_block = False,):
        super(LinearWithActQuant, self).__init__(in_features, out_features, bias)
        self.act_quant_int = act_quant_int
        self.act_quant_fp = act_quant_fp
        self.act_quant_fp_exponent = act_quant_fp_exponent
        self.act_quant_per_block = act_quant_per_block

    def forward(self, x):
        x = x.squeeze(0)
        if self.act_quant_int > 0:
            x = quant_int(x, self.act_quant_int, self.act_quant_per_block)
        return super().forward(x)


def make_actQuant(module, names, name='', 
                act_quant_int = 0, 
                act_quant_fp = 0, 
                act_quant_fp_exponent = 0,
                act_quant_per_block = False,):
    if isinstance(module, LinearWithActQuant):
        return
    for attr in dir(module):
        try:
            tmp = getattr(module, attr)
            name1 = name + '.' + attr if name != '' else attr
            if name1 in names:
                new_module = LinearWithActQuant(tmp.in_features, tmp.out_features, tmp.bias is not None, act_quant_int, act_quant_fp, act_quant_per_block)
                new_module.weight = tmp.weight
                if tmp.bias is not None:
                    new_module.bias = tmp.bias
                setattr(module, attr, new_module)
        except:
            pass
    for name1, child in module.named_children():
        make_actQuant(child, names, name + '.' + name1 if name != '' else name1, act_quant_int, act_quant_fp, act_quant_fp_exponent, act_quant_per_block)

if __name__ == '__main__':
    data = torch.cat([torch.range(1, 16).unsqueeze(0), torch.range(17, 32).unsqueeze(0)], dim=0)
    print(data)
    out = quant_fp(data, 4, 2, per_block=False)
    print(out)
    print(nn.functional.mse_loss(data, out))