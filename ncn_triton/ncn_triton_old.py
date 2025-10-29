import triton
import triton.language as tl
from torch import nn
import torch
import math
import time
from typing import Tuple


from .ncn_triton_utils import _take_slice_
from .group_tensors import extract_group_index_


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

@triton.jit
def leaky_relu_grad(x):
    return tl.where(x >= 0, 1.0, 0.01)


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def tanh_grad(x):
    # Tanh is just a scaled sigmoid
    t = tanh(x)
    return 1 - t * t



# @triton.autotune(
#     [
#         triton.Config(
#             {"BLOCK_X": BLOCK_SIZE_X},
#             num_stages=num_stages,
#             num_warps=num_warps,
#         )
#         for BLOCK_SIZE_X in [1, 2, 4, 8]
#         for num_stages in ([3, 4, 7])
#         for num_warps in [2, 4]
#     ],
#     key=["CTX_LEN",  "EMBED_DIM"],
# )

@triton.jit
def forward_kernel(
        # Pointers to matrices
        x_ptr, yi_ptr, ya_ptr, W_ptr, group_tensor_ptr,
        ctxlen_tensor_ptr,  # this could be int or pointer
        GROUP: tl.constexpr,
        # Matrix dimensions
        BATCH_SIZE: tl.constexpr, NUM_HEADS: tl.constexpr, MAX_CTX_LEN: tl.constexpr, EMBED_DIM: tl.constexpr, HEAD_DIM: tl.constexpr,
        # Meta-parameters
        ACTIVATION: tl.constexpr,
        ALPHA,
        x_recon_ptr
):
    group_index = tl.program_id(axis=0)  # This indicate which block in the sequence length to process
    index_batch_head = tl.program_id(1) # This indicates which head and batch to process. Each program is associated with a single head of a single batch
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    ctxlen = tl.load(ctxlen_tensor_ptr + index_batch)
 
    #group_indices_offsets = index_batch * MAX_CTX_LEN + tl.arange(0, GROUP) + group_index * GROUP
    #group_indices = tl.load(group_tensor_ptr + group_indices_offsets, mask=tl.arange(0, GROUP) + group_index * GROUP < ctxlen).to(tl.int32)
    group_indices = tl.load(group_tensor_ptr + tl.arange(0, GROUP) + group_index * GROUP, mask=tl.arange(0, GROUP) + group_index * GROUP < ctxlen).to(tl.int32)

    xi_offsets = index_batch * MAX_CTX_LEN * EMBED_DIM + group_indices[:, None] * EMBED_DIM + index_head * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
    mask = group_index * GROUP + tl.arange(0, GROUP) < ctxlen + tl.arange(0, GROUP)
    mask = mask[:, None]


    xi = tl.load(x_ptr + xi_offsets, mask=mask)
    xa = tl.zeros_like(xi)   # TODO inits
    #x = tl.load(x_ptr + xi_offsets, mask=mask)

    tl.store(x_recon_ptr+ xi_offsets, xi, mask=mask) 

    # Linear kernel
    offs_Wi = index_head * HEAD_DIM + tl.arange(0, HEAD_DIM)
    offs_Wj = EMBED_DIM + index_head * HEAD_DIM + tl.arange(0, HEAD_DIM)
    Wi = tl.load(W_ptr + offs_Wi)
    Wj = tl.load(W_ptr + offs_Wj)

    if group_index * GROUP + GROUP < ctxlen:
        lo, hi = 0, GROUP
    else:
        lo, hi = 0, ctxlen - group_index * GROUP

    for start_j in range(lo, hi):
        
        xj = _take_slice_(xi, 2, 1, start_j, GROUP, True).broadcast_to(xi.shape)
        #xj = tl.broadcast_to(_take_slice_(xi, 2, 1, start_j, GROUP, True), xi.shape)

        # tl.dot() only work for 2d or 3 matrices M, K, N >=16
        sim = tl.sum(xi * Wi[None, :], axis=1) + tl.sum(xj * Wj[None, :], axis=1)  # (B,)
        T = ALPHA * xi + (1.0-ALPHA) * sim[:, None] * xj  # (B,E)

        # optional: fused activation (while the data is in shared memory)
        if ACTIVATION == "leaky-relu":
            F = leaky_relu(T)
        elif ACTIVATION == "tanh":
            F = tanh(T)
        else:
            F = T

        m = 0.9

        ya = m*xa + (1-m)*F
        yi = m*xi + (1-m)*ya

        xa = ya
        xi = yi
    
    tl.store(yi_ptr + xi_offsets, xi, mask=mask) 
    tl.store(ya_ptr + xi_offsets, xa, mask=mask)




# @triton.autotune(
#     [
#         triton.Config(
#             {},
#             num_stages=num_stages,
#             num_warps=num_warps,
#         )
#         for num_stages in ([3, 4, 7])
#         for num_warps in [2, 4]
#     ],
#     key=["EMBED_DIM"],
# )
@triton.jit
def backward_kernel(
        # Pointers to matrices
        x_ptr, yi_ptr, ya_ptr, W_ptr, 
        dyi_ptr, dya_ptr, dW_ptr, dxi_ptr, dxa_ptr, 
        group_tensor_ptr, 
        ctxlen_tensor_ptr,  # this could be int or pointer
        GROUP: tl.constexpr,
        # Matrix dimensions
        BATCH_SIZE: tl.constexpr, NUM_HEADS: tl.constexpr, MAX_CTX_LEN: tl.constexpr, EMBED_DIM: tl.constexpr, HEAD_DIM: tl.constexpr,
        # Meta-parameters
        ACTIVATION: tl.constexpr,
        ALPHA,
        x_recon_ptr, xa_recon_ptr
):
    
    group_index = tl.program_id(axis=0)  # This indicate which block in the sequence length to process
    index_batch_head = tl.program_id(1) # This indicates which head and batch to process. Each program is associated with a single head of a single batch
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    ctxlen = tl.load(ctxlen_tensor_ptr + index_batch)
 
    #group_indices_offsets = index_batch * MAX_CTX_LEN + tl.arange(0, GROUP) + group_index * GROUP
    #group_indices = tl.load(group_tensor_ptr + group_indices_offsets, mask=tl.arange(0, GROUP) + group_index * GROUP < ctxlen).to(tl.int32)
    group_indices = tl.load(group_tensor_ptr + tl.arange(0, GROUP) + group_index * GROUP, mask=tl.arange(0, GROUP) + group_index * GROUP < ctxlen).to(tl.int32)


    xi_offsets = index_batch * MAX_CTX_LEN * EMBED_DIM + group_indices[:, None] * EMBED_DIM + index_head * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
    mask = group_index * GROUP + tl.arange(0, GROUP) < ctxlen + tl.arange(0, GROUP)
    mask = mask[:, None]

    yi = tl.load(yi_ptr + xi_offsets, mask=mask)
    ya = tl.load(ya_ptr + xi_offsets, mask=mask)

    #x = tl.load(x_ptr + xi_offsets, mask=mask)

    dyi = tl.load(dyi_ptr + xi_offsets, mask=mask)
    dya = tl.load(dya_ptr + xi_offsets, mask=mask)

    # Linear kernel
    offs_Wi = index_head * HEAD_DIM + tl.arange(0, HEAD_DIM)
    offs_Wj = EMBED_DIM + index_head * HEAD_DIM + tl.arange(0, HEAD_DIM)
    Wi = tl.load(W_ptr + offs_Wi)
    Wj = tl.load(W_ptr + offs_Wj)

    dWi = tl.zeros([HEAD_DIM], dtype=tl.float32)
    dWj = tl.zeros([HEAD_DIM], dtype=tl.float32)

    if group_index * GROUP + GROUP < ctxlen:
        lo, hi = 0, GROUP
    else:
        lo, hi = 0, ctxlen - group_index * GROUP
        #import pdb; pdb.set_trace()


    for start_j in range(hi-1, lo-1, -1):
        # Use reversibel trick
        m = 0.9
        xi = (yi - ya * (1.0-m)) / m

        xj = _take_slice_(xi, 2, 1, start_j, GROUP, True).broadcast_to(xi.shape)
        #xj = tl.broadcast_to(_take_slice_(xi, 2, 1, start_j, GROUP, True), xi.shape)

        sim = tl.sum(xi * Wi[None, :], axis=1) + tl.sum(xj * Wj[None, :], axis=1)  # (B,)
        T = ALPHA * xi + (1.0-ALPHA) * sim[:, None] * xj  # (B,E)

        if ACTIVATION == "leaky-relu":
            F = leaky_relu(T)
            grad_activation = leaky_relu_grad(T)
        elif ACTIVATION == "tanh":
            F = tanh(T)
            grad_activation = tanh_grad(T)
        else:
            F = T
            grad_activation = 1.0

        xa = (ya - F * (1.0-m)) / m

        ##########
        # backprop
        
        dyi_dT = (1-m) * (1-m) * (grad_activation ) # (B,E)
        dya_dT = (1-m) * (grad_activation ) # (B,E)

        dWi_group_i = tl.sum(dyi * dyi_dT * xj, axis=1, keep_dims=True) * xi * (1.0-ALPHA)  # dWi from yi
        dWi_group_a = tl.sum(dya * dya_dT * xj, axis=1, keep_dims=True) * xi * (1.0-ALPHA)  # dWi from ya

        dWj_group_i = tl.sum(dyi * dyi_dT * xj, axis=1, keep_dims=True) * xj * (1.0-ALPHA)  # dWj from yi
        dWj_group_a = tl.sum(dya * dya_dT * xj, axis=1, keep_dims=True) * xj * (1.0-ALPHA)  # dWj from ya

        dWi_group = dWi_group_i + dWi_group_a
        dWj_group = dWj_group_i + dWj_group_a

        dWi += tl.sum(dWi_group, axis=0)  # (E,)
        dWj += tl.sum(dWj_group, axis=0)  # (E,)

        # optimized vjp

        # dxi from yi and ya
        dxi_i = dyi*m + ALPHA * dyi * dyi_dT + (1.0 - ALPHA) * ( tl.sum(dyi * dyi_dT * xj, axis=1, keep_dims=True) * Wi[None, :] )  # (B,E)
        dxi_a = ALPHA * dya * dya_dT + (1.0 - ALPHA) * ( tl.sum(dya * dya_dT * xj, axis=1, keep_dims=True) * Wi[None, :] ) # (B,E)
        dxi = dxi_i + dxi_a # (B,E)

        # dxa from yi and ya
        dxa_i = dyi * m * (1-m)  # (B,E)
        dxa_a = dya * m  # (B,E)
        dxa = dxa_i + dxa_a  # (B,E)

        # add dxj to dxi
        # if i==j
        xj_mask = tl.arange(0, GROUP) == start_j
        dxij_i = (1.0 - ALPHA) * ( tl.sum(dyi * dyi_dT * xj, axis=1, keep_dims=True) * (Wj)[None, :]) + (1.0-ALPHA) * dyi * dyi_dT * sim[:, None]  # (B,E)
        dxij_a = (1.0 - ALPHA) * ( tl.sum(dya * dya_dT * xj, axis=1, keep_dims=True) * (Wj)[None, :]) + (1.0-ALPHA) * dya * dya_dT * sim[:, None]  # (B,E)
        dxij = dxij_i + dxij_a # (B,E)
        dxi += tl.where(xj_mask[:, None], dxij, 0.0)

        # else if i!=j
        dxj_i = (1.0 - ALPHA) * ( tl.sum(dyi * dyi_dT * xj, axis=1, keep_dims=True) * Wj[None, :]) + (1.0-ALPHA) * dyi * dyi_dT * sim[:, None]  # (B,E)
        dxj_a = (1.0 - ALPHA) * ( tl.sum(dya * dya_dT * xj, axis=1, keep_dims=True) * Wj[None, :]) + (1.0-ALPHA) * dya * dya_dT * sim[:, None]  # (B,E)
        dxj_group = dxj_i + dxj_a # (B,E)
        dxj = tl.sum(tl.where((xj_mask==False)[:, None], dxj_group, 0.0), axis=0)  # (E,)
        dxi += tl.where(xj_mask[:, None], dxj[None, :], tl.zeros_like(dxi))

        ya = xa
        yi = xi
        dyi = dxi
        dya = dxa


    tl.store(dxi_ptr + xi_offsets, dyi, mask=mask) 
    tl.store(dxa_ptr + xi_offsets, dya, mask=mask) 

    num_groups = tl.cdiv(MAX_CTX_LEN, GROUP)

    dWi_offsets = index_batch * (num_groups * 2*EMBED_DIM) + group_index * (2*EMBED_DIM) + index_head * HEAD_DIM + tl.arange(0, HEAD_DIM)
    dWj_offsets = index_batch * (num_groups * 2*EMBED_DIM) + group_index * (2*EMBED_DIM) + EMBED_DIM + index_head * HEAD_DIM + tl.arange(0, HEAD_DIM)

    tl.store(dW_ptr + dWi_offsets, dWi)
    tl.store(dW_ptr + dWj_offsets, dWj)

    tl.store(x_recon_ptr+ xi_offsets, yi, mask=mask) 
    tl.store(xa_recon_ptr+ xi_offsets, ya, mask=mask) 



class FusedLinearChunkNCNFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, xa, W, group_tensor, ctxlen_tensor, ALPHA, ACTIVATION, NUM_HEADS, CACHE, l):
        # shape constraints
        BATCH_SIZE, MAX_SEQ_LEN, EMBED_DIM = x.shape
        #xa = torch.zeros_like(x)
        yi = torch.empty_like(x)
        ya = torch.empty_like(xa)

        x_recon=torch.zeros_like(yi)

        GROUP = CACHE

        grid = lambda META: (triton.cdiv(MAX_SEQ_LEN, GROUP), BATCH_SIZE*NUM_HEADS, 1)
        
        forward_kernel[grid](
            x_ptr=x, yi_ptr=yi, ya_ptr=ya, W_ptr=W, 
            group_tensor_ptr=group_tensor,ctxlen_tensor_ptr=ctxlen_tensor,
            BATCH_SIZE=BATCH_SIZE, NUM_HEADS=NUM_HEADS, MAX_CTX_LEN=MAX_SEQ_LEN, EMBED_DIM=EMBED_DIM, HEAD_DIM=EMBED_DIM//NUM_HEADS,
            ACTIVATION=ACTIVATION,
            ALPHA=ALPHA,
            GROUP=GROUP,
            x_recon_ptr=x_recon
        )



        ctx.save_for_backward(x, W, yi, ya)
        ctx.ALPHA = ALPHA
        ctx.GROUP = GROUP
        ctx.ACTIVATION = ACTIVATION
        ctx.NUM_HEADS = NUM_HEADS
        ctx.group_tensor = group_tensor
        ctx.ctxlen_tensor = ctxlen_tensor

        return yi, ya



    @staticmethod
    def backward(ctx, dyi, dya):
        x, W, yi, ya  = ctx.saved_tensors
        BATCH_SIZE, MAX_SEQ_LEN, EMBED_DIM = x.shape
        NUM_GROUPS = triton.cdiv(MAX_SEQ_LEN, ctx.GROUP)

        dxi = torch.zeros_like(dyi)
        dxa = torch.zeros_like(dya)

        dW = torch.zeros((BATCH_SIZE, NUM_GROUPS, 2*EMBED_DIM), dtype=W.dtype, device=W.device)

        #grid = lambda META: (BATCH_SIZE, triton.cdiv(CTX_LEN, META["GROUP"]))
        grid = lambda META: (triton.cdiv(MAX_SEQ_LEN, ctx.GROUP), BATCH_SIZE*ctx.NUM_HEADS, 1)

        x_recon=torch.zeros_like(yi)
        xa_recon=torch.zeros_like(yi)

        backward_kernel[grid](
            x_ptr=x, yi_ptr=yi, ya_ptr=ya, W_ptr=W,
            dyi_ptr=dyi, dya_ptr=dya, dW_ptr=dW, dxi_ptr=dxi, dxa_ptr=dxa, 
            group_tensor_ptr=ctx.group_tensor, ctxlen_tensor_ptr=ctx.ctxlen_tensor,
            BATCH_SIZE=BATCH_SIZE, NUM_HEADS=ctx.NUM_HEADS, MAX_CTX_LEN=MAX_SEQ_LEN, EMBED_DIM=EMBED_DIM, HEAD_DIM=EMBED_DIM//ctx.NUM_HEADS,
            ACTIVATION=ctx.ACTIVATION,
            ALPHA=ctx.ALPHA,
            GROUP=ctx.GROUP,
            x_recon_ptr=x_recon,
            xa_recon_ptr=xa_recon,
        )

        #import pdb; pdb.set_trace()
        
        sum_dW = torch.sum(dW, dim=(0, 1))
        return dxi, dxa, sum_dW, None, None, None, None, None, None, None



def fused_chunk_linear_ncn(
    x: torch.Tensor, 
    xa: torch.Tensor, 
    W: torch.Tensor, 
    group_tensor: torch.Tensor, 
    ctxlen_tensor: torch.Tensor, 
    ALPHA: float, 
    ACTIVATION: str,
    NHEAD: int,
    CACHE: int,
    l: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    yi, ya = FusedLinearChunkNCNFunction.apply(
        x, 
        xa,
        W, 
        group_tensor, 
        ctxlen_tensor, 
        ALPHA, 
        ACTIVATION,
        NHEAD,
        CACHE,
        l
    )
    return yi, ya
