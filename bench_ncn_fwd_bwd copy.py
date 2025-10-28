import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Load the CUDA kernel as a python module
ncn_fwd = load(name='ncn_fwd', sources=['main_ncn.cpp', 'ncn_fwd.cu'], extra_cuda_cflags=['-O2'])
ncn_bwd = load(name='ncn_bwd', sources=['main_ncn_bwd.cpp', 'ncn_bwd.cu'], extra_cuda_cflags=['-O2'])


# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 4
n_head = 4
seq_len = 256
embd = 128

alpha = 0.9
n_cache = 32
activation = 0
l = 1


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


print('=== profiling manual ncn ===')


def ref_fwd(X, Xa, W, alpha, activation, n, C, group_idx, idx_within_group):
    B, N, E = X.shape
    Y_, Ya_ = torch.zeros_like(X), torch.zeros_like(Xa)
    for b in range(B):
        
        for c in range(C):
            ce = E // C
            x_bc = X[b, :, c * ce : (c+1) * ce]
            xa_bc = Xa[b, :, c * ce : (c+1) * ce]
            W_ci = W[c * ce : (c+1) * ce]
            W_cj = W[E + c * ce : E + (c+1) * ce]
            n_groups = torch.unique(group_idx)
            idxs = torch.arange(0, N)
            for group in n_groups:
                group_mask = group_idx == group

                idxs = group_mask.nonzero(as_tuple=True)[0] # original positions for group g
                order = idx_within_group[idxs].argsort()       # order within this group's indices
                idxs_sorted = idxs[order]                      # positions sorted inside group
                xi_group = x_bc[idxs_sorted]                        # extracted values in correct order
                xa_group = xa_bc[idxs_sorted]                        # extracted values in correct order

                #xi_group = x_bc[group_mask]
                #indices_group = idx_within_group[group_mask]
                #order = torch.argsort(indices_group)
                #xi_group_ordered = xi_group[order]
                
                xi = xi_group
                xa = xa_group
                
                for j in range(n):
                    #import pdb; pdb.set_trace()

                    
                    xj = xi[j].unsqueeze(0).broadcast_to(xi.shape).detach().clone()  # (n, Ec)
                    xj.requires_grad = False

                    #phi = torch.cat((xi, xj), dim=-1) # (n, 2Ec)
                    Wij =  torch.matmul(xi, W_ci) + torch.matmul(xj, W_cj)

                    T = alpha * xi + (1.0-alpha) * Wij[:, None] * xj  # (n,Ec)

                    F = torch.tanh(T)

                    
                    m = 0.9
                    ya = xa*m + (1-m)*F
                    yi = xi*m + (1-m)*ya

                    #ya = xa + F 
                    #yi = xi + ya

                    xi = yi
                    xa = ya
                
                Y_[b, idxs_sorted, c * ce : (c+1) * ce] += xi
                Ya_[b, idxs_sorted, c * ce : (c+1) * ce] += xa
                

    return Y_, Ya_


dY = torch.randn_like(x)
dYa = torch.randn_like(xa)


indices = torch.arange(0, seq_len)
group_idx = indices // n_cache
idx_within_group = indices % n_cache



with torch.autograd.profiler.profile(use_cuda=True) as prof:
    yi_ref, ya_ref = ref_fwd(x, xa, W, alpha, activation, n_cache, n_head, group_idx, idx_within_group)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

yi_ref.backward(dY)


ref_dx, x.grad = x.grad.clone(), None
ref_dxa, xa.grad = xa.grad.clone(), None
ref_dW, W.grad = W.grad.clone(), None



print('=== profiling minimal flash attention === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    yi, ya = ncn_fwd.forward(x, xa, W, alpha, activation, n_cache, n_head, 1)
    #results = ncn_fwd.forward(x, xa, W, alpha, activation, n_cache, n_head, 1)
    #print(results.shape)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))



with torch.autograd.profiler.profile(use_cuda=True) as prof:
    X, Xa, dX, dXa, dW = ncn_bwd.backward(yi, ya, dY, dYa, W, alpha, activation, n_cache, n_head, 1)
    #results = ncn_fwd.forward(x, xa, W, alpha, activation, n_cache, n_head, 1)
    #print(results.shape)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))


print('yi values sanity check:', torch.allclose(yi_ref, yi, rtol=0, atol=1e-02))
print('ya values sanity check:', torch.allclose(ya_ref, ya, rtol=0, atol=1e-02))


print('xi values sanity check:', torch.allclose(x, X, rtol=0, atol=1e-02))
print('xa values sanity check:', torch.allclose(xa, Xa, rtol=0, atol=1e-02))


print('dxi values sanity check:', torch.allclose(ref_dx, dX, rtol=0, atol=1e-02))
print('dxa values sanity check:', torch.allclose(ref_dxa, dXa, rtol=0, atol=1e-02))

print('dW values sanity check:', torch.allclose(ref_dW, dW.reshape(-1, 2*embd).sum(0), rtol=0, atol=1e-02))

import pdb; pdb.set_trace()