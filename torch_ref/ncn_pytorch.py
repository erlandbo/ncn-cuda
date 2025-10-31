import torch
from torch import nn
from torch.nn import functional as F



def extract_group_index_simple(group_nr, idx_within_group, cache_dim, ctx_len):
    i = group_nr * cache_dim + idx_within_group;
    return i


def group_pos_to_index_v2(N: int, block_size: int, stage: int,
                       group_nr: int, pos_in_group: int,
                       one_based: bool = False) -> int:
    """
    Map (group_nr, pos_in_group) -> original index (1..N) for stage_groups_local_first.
    - N: total indices
    - block_size: size of each group
    - stage: stage number (0 -> local)
    - group_nr: group number (0-based if one_based=False, otherwise 1-based)
    - pos_in_group: position inside group (0-based if one_based=False, otherwise 1-based)
    - one_based: True means group_nr and pos_in_group are 1-based; returned index is 1-based.
    Returns: index in 1..N
    """
    assert N % block_size == 0, "block_size must divide N"
    m = N // block_size
    if one_based:
        b = group_nr - 1
        o = pos_in_group - 1
    else:
        b = group_nr
        o = pos_in_group
    

    assert 0 <= b < m, "group_nr out of range"
    assert all((0 <= o) & (o < block_size)), "pos_in_group out of range"

    s = 0 if stage == 0 else (1 << (stage - 1)) % m
    block = (b + o * s) % m
    idx = block * block_size + o
    return idx



def ref_naive_fwd(X, Xa, W, alpha, activation, n, C, module_l):
    B, N, E = X.shape
    Y_ = torch.zeros_like(X)
    Ya_ = torch.zeros_like(Xa)
    # Y_, Ya_ = torch.zeros(B,N,E, device="cuda").requires_grad_(), torch.zeros(B,N,E, device="cuda").requires_grad_()
    for b in range(B):
        
        for c in range(C):
            ce = E // C
            c = int(c)
            ce = int(ce)
            E = int(E)
            b = int(b)
            x_bc = X[b, :, c * ce : (c+1) * ce]
            xa_bc = Xa[b, :, c * ce : (c+1) * ce]
            W_ci = W[c * ce : (c+1) * ce]
            W_cj = W[E + c * ce : E + (c+1) * ce]
            for group_nr in range(N//n):
                idx_within_group = torch.arange(0, n, dtype=torch.long)
                #xi_idxes = extract_group_index_simple(group_nr, idx_within_group, n, N)
                xi_idxes = group_pos_to_index_v2(N, n, module_l, group_nr, idx_within_group, one_based=False)
                xi_idxes = xi_idxes.to(torch.long)

                xi_group = x_bc[xi_idxes]                        # extracted values in correct order
                xa_group = xa_bc[xi_idxes]                        # extracted values in correct order

                xi = xi_group
                xa = xa_group
                
                for j in range(n):
                    j = int(j)

                    xj = xi[j].unsqueeze(0).broadcast_to(xi.shape)#.detach().clone()  # (n, Ec)

                    #phi = torch.cat((xi, xj), dim=-1) # (n, 2Ec)
                    #import pdb; pdb.set_trace()
                    Wij =  torch.matmul(xi, W_ci) + torch.matmul(xj, W_cj)

                    T = alpha * xi + (1.0-alpha) * Wij[:, None] * xj  # (n,Ec)
                    #T = alpha * xi + (1.0-alpha) * Wij.unsqueeze(1).broadcast_to(xj.shape) * xj  # (n,Ec)
                    #T = alpha * xi + (1.0-alpha) * Wij.unsqueeze(1) * xj  # (n,Ec)

                    F = torch.tanh(T)
                    
                    m = 0.9
                    ya = xa*m + (1-m)*F
                    yi = xi*m + (1-m)*ya

                    xi = yi
                    xa = ya
                
                Y_[b, xi_idxes, c * ce : (c+1) * ce] += xi
                Ya_[b, xi_idxes, c * ce : (c+1) * ce] += xa
    
    return Y_, Ya_



class RefTorchNaive(nn.Module):
    def __init__(
        self,
        alpha,
        activation,
        num_layers,
        cache,
        n_kernel_feats,
        weights
    ):
        super().__init__()
        self.weights = weights
        self.alpha = alpha
        self.activation = activation
        self.num_layers = num_layers
        self.n_kernel_feats = n_kernel_feats
        self.cache = cache

    def forward(self, x, xa):
        for l in range(len(self.weights)):
            x, xa = ref_naive_fwd(x, xa, self.weights[l], self.alpha, self.activation, self.cache, self.n_kernel_feats, l)
        return x, xa
