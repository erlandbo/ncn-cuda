import math
import torch
from typing import Tuple
from torch.utils.cpp_extension import load


ncn_cuda_module = load(name='ncn_cuda', sources=['ncn_cuda/ncn_cuda.cpp', 'ncn_cuda/ncn_fwd_cuda_kernel.cu', 'ncn_cuda/ncn_bwd_cuda_kernel.cu'], extra_cuda_cflags=['-O2'])


class NCNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, xa, W, alpha, activation, n_cache, n_head, module_l):
        yi, ya = ncn_cuda_module.forward(
            x, 
            xa, 
            W, 
            alpha, 
            activation, 
            n_cache, 
            n_head, 
            module_l
        )
        ctx.save_for_backward(yi, ya, W)
        ctx.alpha = alpha
        ctx.activation = {"tanh": 0, "relu": 1, "sigmoid": 2, "linear": -1}[activation]
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

        dW_reduced = dW.flatten(0,1).sum(0)

        return dx, dxa, dW_reduced, None, None, None, None, None


def fused_ncn_cuda_v1(
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


class NCN(torch.nn.Module):
    def __init__(self, alpha, activation, n_cache, n_head, module_l, embed_dim):
        super(NCN, self).__init__()
        self.alpha = alpha
        self.activation = activation
        self.n_cache = n_cache
        self.n_head = n_head
        self.module_l = module_l
        self.embed_dim = embed_dim
        self.W = torch.nn.Parameter(
            torch.empty(2 * embed_dim))
        self.bias = torch.nn.Parameter(torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embed_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, x, xa):
        return NCNFunction.apply(x, xa, self.W, self.alpha, self.activation, self.n_cache, self.n_head, self.module_l)
    