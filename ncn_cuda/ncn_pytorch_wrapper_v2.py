import math
import torch
from torch import nn
from typing import Tuple

import ncn_cuda_module


class NCNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, xa, W, alpha, activation, n_cache, n_head, module_l):
        yi, ya = ncn_cuda_module.forward(
            x, 
            xa, 
            W.data, 
            alpha, 
            activation, 
            n_cache, 
            n_head, 
            module_l
        )
        ctx.save_for_backward(yi, ya, W)
        ctx.alpha = alpha
        ctx.activation = activation
        ctx.n_cache = n_cache
        ctx.n_head = n_head
        ctx.module_l = module_l

        return yi, ya

    @staticmethod
    def backward(ctx, dyi, dya):
        (yi, ya, W) = ctx.saved_tensors

        x, xa, dx, dxa, dW = ncn_cuda_module.backward(
            yi, 
            ya, 
            dyi, 
            dya, 
            W, 
            ctx.alpha, 
            ctx.activation, 
            ctx.n_cache, 
            ctx.n_head, 
            ctx.module_l
        )

        sum_dW = torch.sum(dW, dim=(0, 1))

        return dx, dxa, sum_dW, None, None, None, None, None


def fused_ncn_cuda_v2(
    x: torch.Tensor,
    xa: torch.Tensor,
    W: torch.Tensor,
    alpha: float, 
    activation: int, 
    n_cache: int, 
    n_head: int, 
    module_l: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    yi, ya = NCNFunction.apply(
        x, 
        xa, 
        W, 
        alpha, 
        activation, 
        n_cache, 
        n_head, 
        module_l
    )
    return yi, ya



class NCNModuleCuda(nn.Module):
    def __init__(self, alpha, activation, n_cache, n_head, module_nr, embed_dim):
        super(NCNModuleCuda, self).__init__()
        self.alpha = alpha
        self.activation = activation
        self.n_cache = n_cache
        self.n_head = n_head
        self.module_nr = module_nr
        self.weight = torch.nn.Parameter(torch.ones(2*embed_dim).normal_(mean=0.0, std=1/(2.0 * embed_dim)), requires_grad=True)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embed_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, x, xa):
        y, ya = fused_ncn_cuda_v2(
                x, 
                xa, 
                self.weight, 
                self.alpha,
                self.activation, 
                self.n_head, 
                self.n_cache, 
                self.module_nr
        )
        return y, ya



class NCNNetCuda(nn.Module):
    def __init__(self, alpha, activation, n_cache, n_head, num_modules, embed_dim):
        super(NCNNetCuda, self).__init__()
        self.num_modules = num_modules
        activation_nr = {"tanh": 0, "relu": 1, "sigmoid": 2, "linear": -1}[activation]
        self.ncn_modules = nn.ModuleList([NCNModuleCuda( alpha, activation_nr, n_cache, n_head, module_nr, embed_dim) for module_nr in range(num_modules)])

    def forward(self, x, xa):
        for module_nr in range(self.num_modules):
            x, xa = self.ncn_modules[module_nr](x, xa)
        return x, xa
    


class NCNNetTestCuda(nn.Module):
    def __init__(self, alpha, activation, n_cache, n_head, num_modules, weights):
        super(NCNNetTestCuda, self).__init__()
        self.num_modules = num_modules
        self.alpha = alpha
        self.activation_nr = {"tanh": 0, "relu": 1, "sigmoid": 2, "linear": -1}[activation]
        self.n_cache = n_cache
        self.n_head = n_head
        self.weights = weights


    def forward(self, x, xa):
        for module_nr in range(self.num_modules):
            x, xa = fused_ncn_cuda_v2(
                x, 
                xa, 
                self.weights[module_nr], 
                self.alpha,
                self.activation_nr, 
                self.n_cache, 
                self.n_head, 
                module_nr
        )
        return x, xa
    