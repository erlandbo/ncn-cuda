import triton
import triton.language as tl
import torch
import math



def extract_group_index(i , l , strides , n):
    # strides [l] equals n ** l.
    # lower part : digits 0 ... l -1
    lower = i % strides[l]
    # the digit at position l
    index_within_group = (i // strides[l]) % n
    # digits above position l , shifted down one place
    upper = i // (n * strides[l])
    # reassemble the group number without i_l
    group_number = lower + upper * strides[l]
    return index_within_group , group_number



@triton.jit
def extract_group_index_kernel( i_ptr , index_ptr , group_ptr , strides_ptr , l: int, n: tl.constexpr , size : tl.constexpr , BLOCK_SIZE : tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    idx = offset + tl.arange (0, BLOCK_SIZE)
    mask = idx < size
    i_val = tl.load( i_ptr + idx , mask = mask ).to(tl.int32)
    l_val = l
    stride = tl.load( strides_ptr + l_val).to(tl.int32)
    lower = i_val % stride
    index_within_group = ( i_val // stride ) % n
    upper = i_val // (n * stride )
    group_number = lower + upper * stride
    tl.store ( index_ptr + idx , index_within_group , mask = mask )
    tl.store ( group_ptr + idx , group_number , mask = mask )



def extract_group_index_(l, N, n):
    K = math.log(N, n)

    i_tensor = torch.arange(0, N).cuda()
    strides_tensor = n**torch.arange(0, K).to(i_tensor.dtype).cuda()

    index_within_group_tensor = torch.empty_like( i_tensor )
    group_number_tensor = torch.empty_like( i_tensor )
    BLOCK_SIZE = 2048
    grid = lambda meta : ( math.ceil ( N / meta["BLOCK_SIZE"]) ,)
    extract_group_index_kernel [ grid ](
        i_tensor , index_within_group_tensor ,
        group_number_tensor , strides_tensor ,
        l, n , N , BLOCK_SIZE
    )
    return index_within_group_tensor , group_number_tensor

