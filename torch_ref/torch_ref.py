import torch
from torch.nn import functional as F


def extract_group_index_simple(group_nr, idx_within_group, cache_dim, ctx_len):
    i = group_nr * cache_dim + idx_within_group;
    return i


def ref_naive_fwd(X, Xa, W, alpha, activation, n, C):
    B, N, E = X.shape
    Y_, Ya_ = torch.zeros_like(X), torch.zeros_like(Xa)
    for b in range(B):
        
        for c in range(C):
            ce = E // C
            x_bc = X[b, :, c * ce : (c+1) * ce]
            xa_bc = Xa[b, :, c * ce : (c+1) * ce]
            W_ci = W[c * ce : (c+1) * ce]
            W_cj = W[E + c * ce : E + (c+1) * ce]
            for group_nr in range(N//n):
                idx_within_group = torch.arange(0, n)
                xi_idxes = extract_group_index_simple(group_nr, idx_within_group, n, N)


                xi_group = x_bc[xi_idxes]                        # extracted values in correct order
                xa_group = xa_bc[xi_idxes]                        # extracted values in correct order

                xi = xi_group
                xa = xa_group
                
                for j in range(n):

                    xj = xi[j].unsqueeze(0).broadcast_to(xi.shape)#.detach().clone()  # (n, Ec)

                    #phi = torch.cat((xi, xj), dim=-1) # (n, 2Ec)
                    Wij =  torch.matmul(xi, W_ci) + torch.matmul(xj, W_cj)

                    T = alpha * xi + (1.0-alpha) * Wij[:, None] * xj  # (n,Ec)

                    F = torch.tanh(T)
                    
                    m = 0.9
                    ya = xa*m + (1-m)*F
                    yi = xi*m + (1-m)*ya

                    xi = yi
                    xa = ya
                
                Y_[b, xi_idxes, c * ce : (c+1) * ce] += xi
                Ya_[b, xi_idxes, c * ce : (c+1) * ce] += xa
                

    return Y_, Ya_
