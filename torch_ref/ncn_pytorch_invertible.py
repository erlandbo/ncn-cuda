import torch
from torch import nn

from torch.nn import functional as F
import math


# https://github.com/tbung/pytorch-revnet/blob/master/revnet/revnet.py

class NCNFunctionPytorch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xi, xa, Wi, Wj, ALPHA, ACTIVATION, NUM_HEADS, CACHE, MODULE_L):
        # shape constraints
        #BATCH_SIZE, MAX_SEQ_LEN, EMBED_DIM = xi.shape
        yi = torch.empty_like(xi)
        ya = torch.empty_like(xa)
        xi_init = xi.detach().clone()
        xa_init = xa.detach().clone()

        GROUP = CACHE

        x_values = []
        xa_values = []

        with torch.no_grad():

            for start_j in range(0, GROUP):

                x_values.append(xi)
                xa_values.append(xa)
                
                xj = xi[start_j].unsqueeze(0).broadcast_to(xi.shape)

                sim = torch.sum(xi * Wi[None, :], axis=1) + torch.sum(xj * Wj[None, :], axis=1)  # (B,)
                T = ALPHA * xi + (1.0-ALPHA) * sim[:, None] * xj  # (B,E)

                # optional: fused activation (while the data is in shared memory)
                if ACTIVATION == "leaky-relu":
                    F = torch.leaky_relu(T)
                elif ACTIVATION == "tanh":
                    F = torch.tanh(T)
                else:
                    F = T

                m = 0.9

                ya = m*xa + (1-m)*F
                yi = m*xi + (1-m)*ya

                xa = ya
                xi = yi
        
            yi = xi
            ya = xa

        ctx.save_for_backward(yi, ya, Wi, Wj, xi_init, xa_init)
        ctx.ALPHA = ALPHA
        ctx.GROUP = GROUP
        ctx.ACTIVATION = ACTIVATION
        ctx.NUM_HEADS = NUM_HEADS
        ctx.MODULE_L = MODULE_L
        ctx.x_values = x_values
        ctx.xa_values = xa_values

        return yi, ya



    @staticmethod
    def backward(ctx, dyi, dya):
        yi, ya, Wi, Wj, xi_init, xa_init  = ctx.saved_tensors
        #BATCH_SIZE, MAX_SEQ_LEN, EMBED_DIM = yi.shape
        #NUM_GROUPS = MAX_SEQ_LEN, ctx.GROUP

        dxi = torch.zeros_like(dyi)
        dxa = torch.zeros_like(dya)

        dWi = torch.zeros_like(Wi)
        dWj = torch.zeros_like(Wj)

        dxi, dxa, dWi, dWj, xi_rec, xa_recon = NCNFunctionPytorch.grad(dyi, dya, yi, ya, Wi, Wj, ctx.GROUP, ctx.ALPHA, ctx.ACTIVATION, ctx.x_values, ctx.xa_values)
        #import pdb; pdb.set_trace()
        return dxi, dxa, dWi, dWj, None, None, None, None, None


    @staticmethod
    def invert(yi, ya, Wi, Wj, GROUP, ALPHA, ACTIVATION, start_j):
        xi = torch.empty_like(yi)
        xa = torch.empty_like(ya)
        lo, hi = 0, GROUP

        m = 0.9

        with torch.no_grad():
            # Use reversibel trick
            xi = (yi - ya * (1.0-m)) / m

            xj = xi[start_j].unsqueeze(0).broadcast_to(xi.shape)

            sim = torch.sum(xi * Wi[None, :], axis=1) + torch.sum(xj * Wj[None, :], axis=1)  # (B,)
            T = ALPHA * xi + (1.0-ALPHA) * sim[:, None] * xj  # (B,E)

            if ACTIVATION == "leaky-relu":
                F = torch.leaky_relu(T)
            elif ACTIVATION == "tanh":
                F = torch.tanh(T)
            else:
                F = T

            xa = (ya - F * (1.0-m)) / m

        return xi, xa



    @staticmethod
    def grad(dyi, dya, yi, ya, Wi, Wj, GROUP, ALPHA, ACTIVATION, x_values, xa_values):

        dxi = torch.zeros_like(dyi)
        dxa = torch.zeros_like(dya)

        dWi = torch.zeros_like(Wi)
        dWj = torch.zeros_like(Wj)
        
        m = 0.9

        lo, hi = 0, GROUP

        for start_j in range(hi-1, -1, -1):
            #xi_0, xa_0 = NCNFunctionPytorch.invert(yi, ya, Wi, Wj, GROUP, ALPHA, ACTIVATION, start_j)
            xi_0, xa_0 = x_values[start_j], xa_values[start_j]

            with torch.enable_grad():
                xi = xi_0.detach().requires_grad_(True)
                xa = xa_0.detach().requires_grad_(True)
                Wi = Wi.detach().requires_grad_(True)
                Wj = Wj.detach().requires_grad_(True)

                xj = xi[start_j].unsqueeze(0).broadcast_to(xi.shape)

                sim = torch.sum(xi * Wi[None, :], axis=1) + torch.sum(xj * Wj[None, :], axis=1)  # (B,)
                T = ALPHA * xi + (1.0-ALPHA) * sim[:, None] * xj  # (B,E)

                # optional: fused activation (while the data is in shared memory)
                if ACTIVATION == "leaky-relu":
                    F = torch.leaky_relu(T)
                elif ACTIVATION == "tanh":
                    F = torch.tanh(T)
                else:
                    F = T

                m = 0.9

                #ya = m*xa + (1-m)*F
                #yi = m*xi + (1-m)*ya

                #dxi = dL/dyi dyi/dxi + dL/dya dya/dxi = dL/dyi (dm*xi/dxi + d(1-m)*ya/dxi) + dL/dya (dm*xa/dxi + d(1-m)F/dxi) 
                #dxa = dL/dyi dyi/dxa + dL/dya dya/dxa = dL/dyi (dm*xi/dxa + d(1-m)*ya/dxa) + dL/dya (dm*xa/dxa + d(1-m)F/dxa) 
                
                #import pdb; pdb.set_trace()
                
                #F.backward()
                ddF_yi = torch.autograd.grad(F, (xi, Wi, Wj), dyi, retain_graph=True)  # dL/dyi dyi/F dF/dx
                ddF_ya = torch.autograd.grad(F, (xi, Wi, Wj), dya)

                dWi += (1.0-m) * (1.0-m) * ddF_yi[1] + (1.0-m) * ddF_ya[1]
                dWj += (1.0-m) * (1.0-m) * ddF_yi[2] + (1.0-m) * ddF_ya[2]


                dxi = m * dyi + (1.0-m) * (1.0-m) * ddF_yi[0] + (1.0-m) * ddF_ya[0]
                dxa = dyi * (1.0-m) * m + m * dya
                #dxa =  (1.0-m) * dyi * m + dya * m
                #dxi = m * dyi + (1.0-m) * (1.0-m) * dyi * xi.grad + dya * (1.0-m) * xi.grad
                
                #dxj_i = (1.0-m) * (1.0-m) * xj.grad 
                #dxj_a = (1.0-m) * xj.grad 



                #vjps_ya = torch.autograd.grad(F, (xi, xa, Wi, Wj), dyi + dya)
                #vjps_ya = torch.autograd.grad(ya, (xi, xa, Wi, Wj), dya)
                #vjps_ya = torch.autograd.grad(ya, (xi, xa, Wi, Wj), dya)

                #vjps_yi = torch.autograd.grad(yi, (xi, xa, Wi, Wj), dyi, retain_graph=True)

                #import pdb; pdb.set_trace()
                #dxi = m * dyi + (1-m) * vjps_ya[0] + vjps_ya[0]  #vjps_yi[0] + vjps_ya[0]
                #dxa = (1-m) * vjps_ya[1] + vjps_ya[1] #vjps_yi[1] + vjps_ya[1]
                #dWi += (1-m) * vjps_ya[2] + vjps_ya[2] # vjps_yi[2] + vjps_ya[2]
                #dWj += (1-m) * vjps_ya[3] + vjps_ya[3] #vjps_yi[3] + vjps_ya[3]

                ya = xa.detach()
                yi = xi.detach()

                dyi = dxi.detach()
                dya = dxa.detach()
        
            yi = xi.detach()
            ya = xa.detach()


        return dxi, dxa, dWi, dWj, xi_0, xa_0


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



class RefInvTorchNaive(nn.Module):
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
            x, xa = self.foward_loop(x, xa, self.weights[l], l)
        return x, xa


    def foward_loop(self, X, Xa, W, module_l):
        B, N, E = X.shape
        Y_ = torch.zeros_like(X)
        Ya_ = torch.zeros_like(Xa)
        for b in range(B):
            for c in range(self.n_kernel_feats):
                ce = E // self.n_kernel_feats
                c = int(c)
                ce = int(ce)
                E = int(E)
                b = int(b)
                x_bc = X[b, :, c * ce : (c+1) * ce]
                xa_bc = Xa[b, :, c * ce : (c+1) * ce]
                W_ci = W[c * ce : (c+1) * ce]
                W_cj = W[E + c * ce : E + (c+1) * ce]
                for group_nr in range(N//self.cache):
                    idx_within_group = torch.arange(0, self.cache, dtype=torch.long)
                    xi_idxes = group_pos_to_index_v2(N, self.cache, module_l, group_nr, idx_within_group, one_based=False)
                    xi_idxes = xi_idxes.to(torch.long)

                    xi_group = x_bc[xi_idxes]                        # extracted values in correct order
                    xa_group = xa_bc[xi_idxes]                        # extracted values in correct order

                    xi = xi_group
                    xa = xa_group

                    yi, ya = NCNFunctionPytorch.apply(xi, xa, W_ci, W_cj, self.alpha, self.activation,self.n_kernel_feats, self.cache, module_l)
                    xi, xa = yi, ya

                    Y_[b, xi_idxes, c * ce : (c+1) * ce] += xi
                    Ya_[b, xi_idxes, c * ce : (c+1) * ce] += xa
        
        return Y_, Ya_



if __name__ == "__main__":

    batch_size = 4
    n_head = 4
    seq_len = 256
    embd = 128

    alpha = 0.9
    n_cache = 32
    activation = "tanh"

    max_l = int(math.log2(seq_len // n_cache))
    #max_l = 1
    print("max_l", max_l)

    x = (
        torch.ones(
            (batch_size, seq_len, embd), dtype=torch.float32, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_().cuda()
    )

    xa = (
        torch.zeros(
            (batch_size, seq_len, embd), dtype=torch.float32, device="cuda"
        ) .requires_grad_().cuda()
        
    )

    #Ws = torch.nn.ParameterList([
    #    torch.nn.Parameter(torch.ones(2*embd).normal_(mean=0.0, std=0.5), requires_grad=True).cuda() for _ in range(max_l)
    #])
    Ws = [
        torch.ones(2*embd, dtype=torch.float32, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_().cuda() for _ in range(max_l)
    ]



    dyi = torch.randn_like(x)
    dya = torch.randn_like(xa)  #torch.randn_like(xa)



    net = RefTorchNaive(alpha=alpha, activation=activation, num_layers=max_l, cache=n_cache, n_kernel_feats=n_head, weights=Ws)


    dyi = torch.randn_like(x)
    dya = torch.randn_like(xa)  #torch.randn_like(xa)



    torch_naive_yi, torch_naive_ya = net(x, xa)

    torch_naive_yi.backward(dyi)

    import pdb; pdb.set_trace()