import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

from ncn_triton.ncn_triton import fused_chunk_linear_ncn
from torch_ref.torch_ref import ref_naive_fwd


import time

start_time = time.time()

# Load the CUDA kernel as a python module
#ncn_fwd = load(name='ncn_fwd', sources=['ncn_cuda/main_ncn_fwd.cpp', 'ncn_cuda/ncn_fwd.cu'], extra_cuda_cflags=['-O2'])
#ncn_bwd = load(name='ncn_bwd', sources=['ncn_cuda/main_ncn_bwd.cpp', 'ncn_cuda/ncn_bwd.cu'], extra_cuda_cflags=['-O2'])
ncn_cuda = load(name='ncn_cuda', sources=['ncn_cuda/ncn_cuda.cpp', 'ncn_cuda/ncn_fwd_cuda_kernel.cu', 'ncn_cuda/ncn_bwd_cuda_kernel.cu'], extra_cuda_cflags=['-O2'])


# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 4
n_head = 4
seq_len = 256
embd = 128

alpha = 0.9
n_cache = 32
activation = 0
module_l = 1


x = (
    torch.ones(
        (batch_size, seq_len, embd), dtype=torch.float32, device="cuda"
    )
    .normal_(mean=0.0, std=0.5)
    .requires_grad_()
)

xa = (
    torch.zeros(
        (batch_size, seq_len, embd), dtype=torch.float32, device="cuda"
    )
    .requires_grad_()
)

W = (
    torch.ones(
        (2 * embd), dtype=torch.float32, device="cuda"
    )
    .normal_(mean=0.0, std=0.5)
    .requires_grad_()
)



dyi = torch.randn_like(x)
dya = torch.randn_like(xa) #torch.randn_like(xa)


torch_naive_yi, torch_naive_ya = ref_naive_fwd(x, xa, W, alpha, activation, n_cache, n_head)

torch_naive_yi.backward(dyi, retain_graph=True)
torch_naive_ya.backward(dya)


torch_naive_dx, x.grad = x.grad.clone(), None
torch_naive_dxa, xa.grad = xa.grad.clone(), None
torch_naive_dW, W.grad = W.grad.clone(), None


#cpp_yi, cpp_ya = ncn_fwd.forward(x, xa, W, alpha, activation, n_cache, n_head, module_l)
cpp_yi, cpp_ya = ncn_cuda.forward(x, xa, W, alpha, activation, n_cache, n_head, module_l)

#cpp_x, cpp_xa, cpp_dx, cpp_dxa, cpp_dW = ncn_bwd.backward(cpp_yi, cpp_ya, dyi, dya, W, alpha, activation, n_cache, n_head, module_l)
cpp_x, cpp_xa, cpp_dx, cpp_dxa, cpp_dW = ncn_cuda.backward(cpp_yi, cpp_ya, dyi, dya, W, alpha, activation, n_cache, n_head, module_l)


print('yi values sanity check:', torch.allclose(torch_naive_yi, cpp_yi, rtol=0, atol=1e-02))
print('ya values sanity check:', torch.allclose(torch_naive_ya, cpp_ya, rtol=0, atol=1e-02))

print('dxi values sanity check:', torch.allclose(torch_naive_dx, cpp_dx, rtol=0, atol=1e-02))
print('dxa values sanity check:', torch.allclose(torch_naive_dxa, cpp_dxa, rtol=0, atol=1e-02))

print('dW values sanity check:', torch.allclose(torch_naive_dW, cpp_dW.flatten(0,1).sum(0), rtol=0, atol=1e-02))

print(time.time()- start_time)