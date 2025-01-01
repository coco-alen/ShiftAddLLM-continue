import numpy as np
import torch
import torch.nn as nn

class LinearWithActQuant(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 act_quant_int = 0, 
                 act_quant_fp = 0, 
                 act_quant_per_block = False,):
        super(LinearWithActQuant, self).__init__(in_features, out_features, bias)
        self.act_quant_int = act_quant_int
        self.act_quant_fp = act_quant_fp
        self.act_quant_per_block = act_quant_per_block

    def forward(self, x):
        print(f"act_quant_int: {self.act_quant_int}")
        return super().forward(x)


def make_actQuant(module, names, name='', 
                act_quant_int = 0, 
                act_quant_fp = 0, 
                act_quant_per_block = False,):
    if isinstance(module, LinearWithActQuant):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            new_module = LinearWithActQuant(tmp.in_features, tmp.out_features, tmp.bias is not None, act_quant_int, act_quant_fp, act_quant_per_block)
            new_module.weight = tmp.weight
            if tmp.bias is not None:
                new_module.bias = tmp.bias
            setattr(module, attr, new_module)
    for name1, child in module.named_children():
        make_actQuant(child, names, name + '.' + name1 if name != '' else name1, act_quant_int, act_quant_fp, act_quant_per_block)
