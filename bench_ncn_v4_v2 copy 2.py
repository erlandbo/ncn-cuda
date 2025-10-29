import torch
from ncn_triton.ncn_triton import NCNNetTestTriton
from torch_ref.torch_ref import RefTorchNaive
#from ncn_cuda.ncn_pytorch_wrapper_v1 import fused_ncn_cuda_v1
from ncn_cuda.ncn_pytorch_wrapper_v2 import NCNNetTestCuda

import time
import math

start_time = time.time()

# Load the CUDA kernel as a python module
#ncn_fwd = load(name='ncn_fwd', sources=['ncn_cuda/main_ncn_fwd.cpp', 'ncn_cuda/ncn_fwd.cu'], extra_cuda_cflags=['-O2'])
#ncn_bwd = load(name='ncn_bwd', sources=['ncn_cuda/main_ncn_bwd.cpp', 'ncn_cuda/ncn_bwd.cu'], extra_cuda_cflags=['-O2'])
#ncn_cuda = load(name='ncn_cuda', sources=['ncn_cuda/ncn_cuda.cpp', 'ncn_cuda/ncn_fwd_cuda_kernel.cu', 'ncn_cuda/ncn_bwd_cuda_kernel.cu'], extra_cuda_cflags=['-O2'])

#import ncn_cuda_module


# Use small model params, otherwise slower than manual attention. See caveats in README.
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
    .requires_grad_()
)

xa = (
    torch.zeros(
        (batch_size, seq_len, embd), dtype=torch.float32, device="cuda"
    )
    .requires_grad_()
)

Ws = torch.nn.ParameterList([
    torch.nn.Parameter(torch.ones(2*embd).normal_(mean=0.0, std=0.5), requires_grad=True).cuda() for _ in range(max_l)
])



dyi = torch.randn_like(x)
dya = torch.randn_like(xa) #torch.randn_like(xa)


ref_naive_net = RefTorchNaive(alpha, activation, max_l, n_cache, n_head, Ws).cuda()
start_time_ref = time.time()


torch_naive_yi, torch_naive_ya = ref_naive_net(x, xa)


torch_naive_yi.backward(dyi, retain_graph=True)
torch_naive_ya.backward(dya)

torch_naive_dWs = []

for i in range(max_l-1, -1, -1):
    torch_naive_dW = ref_naive_net.weights[i].grad.clone()
    ref_naive_net.weights[i].grad = None
    torch_naive_dWs.append(torch_naive_dW)


torch_naive_dx, x.grad = x.grad.clone(), None
torch_naive_dxa, xa.grad = xa.grad.clone(), None


end_time_ref = time.time()



start_time_cuda = time.time()


ncn_net_cuda = NCNNetTestCuda(alpha, activation, n_cache, n_head, max_l, Ws).cuda()
cpp_yi, cpp_ya = ncn_net_cuda(x, xa)


cpp_yi.backward(dyi, retain_graph=True)
cpp_ya.backward(dya)

cpp_dWs = []

for i in range(max_l):
    cpp_dW = ncn_net_cuda.weights[i].grad.clone()
    ncn_net_cuda.weights[i].grad = None
    cpp_dWs.append(cpp_dW)

cpp_dx, x.grad = x.grad.clone(), None
cpp_dxa, xa.grad = xa.grad.clone(), None



end_time_cuda = time.time()



start_time_triton = time.time()

ncn_net_triton = NCNNetTestTriton(alpha, activation, n_cache, n_head, max_l, Ws).cuda()
tri_yi, tri_ya = ncn_net_triton(x, xa)

tri_yi.backward(dyi, retain_graph=True)
tri_ya.backward(dya)


tri_dWs = []

for i in range(max_l):
    tri_dW = ncn_net_triton.weights[i].grad.clone()
    ncn_net_triton.weights[i].grad = None
    tri_dWs.append(tri_dW)

tri_dx, x.grad = x.grad.clone(), None
tri_dxa, xa.grad = xa.grad.clone(), None



end_time_triton = time.time()



try:
    torch.testing.assert_close(torch_naive_yi, cpp_yi, rtol=0, atol=1e-02)
    print('yi values sanity check pytorch, cuda:', torch.testing.assert_close(torch_naive_yi, cpp_yi, rtol=0, atol=1e-02))
except AssertionError as e:
    print("yi values sanity check pytorch, cuda:", e)

try:
    torch.testing.assert_close(torch_naive_ya, cpp_ya, rtol=0, atol=1e-02)
    print('ya values sanity check pytorch, cuda:', torch.testing.assert_close(torch_naive_ya, cpp_ya, rtol=0, atol=1e-02))
except AssertionError as e:
    print('ya values sanity check pytorch, cuda:' ,e)

try:
    torch.testing.assert_close(torch_naive_dx, cpp_dx, rtol=0, atol=1e-02)
    print('dxi values sanity check pytorch, cuda:', torch.testing.assert_close(torch_naive_dx, cpp_dx, rtol=0, atol=1e-02))
except AssertionError as e:
    print('dxi values sanity check pytorch, cuda:', e)

try:
    torch.testing.assert_close(torch_naive_dxa, cpp_dxa, rtol=0, atol=1e-02)
    print('dxa values sanity check pytorch, cuda:', torch.testing.assert_close(torch_naive_dxa, cpp_dxa, rtol=0, atol=1e-02))
except AssertionError as e:
    print('dxa values sanity check pytorch, cuda:',e)

for i in range(max_l):
    try:
        torch.testing.assert_close(torch_naive_dWs[i], cpp_dWs[i], rtol=0, atol=1e-02)
        print(i ,'dW  values sanity check pytorch, cuda:', torch.testing.assert_close(torch_naive_dWs[i], cpp_dWs[i], rtol=0, atol=1e-02))
    except AssertionError as e:
        print(i, 'dW values sanity check pytorch, cuda:',e)





try:
    torch.testing.assert_close(torch_naive_yi, tri_yi, rtol=0, atol=1e-02)
    print('yi values sanity check pytorch, triton:', torch.testing.assert_close(torch_naive_yi, tri_yi, rtol=0, atol=1e-02))
except AssertionError as e:
    print('yi values sanity check pytorch, triton:',e)

try:
    torch.testing.assert_close(torch_naive_ya, tri_ya, rtol=0, atol=1e-02)
    print('ya values sanity check pytorch, triton:', torch.testing.assert_close(torch_naive_ya, tri_ya, rtol=0, atol=1e-02))
except AssertionError as e:
    print('ya values sanity check pytorch, triton:',e)

try:
    torch.testing.assert_close(torch_naive_dx, tri_dx, rtol=0, atol=1e-02)
    print('dxi values sanity check pytorch, triton:', torch.testing.assert_close(torch_naive_dx, tri_dx, rtol=0, atol=1e-02))
except AssertionError as e:
    print('dxi values sanity check pytorch, triton:',e)

try:
    torch.testing.assert_close(torch_naive_dxa, tri_dxa, rtol=0, atol=1e-02)
    print('dxa values sanity check pytorch, triton:', torch.testing.assert_close(torch_naive_dxa, tri_dxa, rtol=0, atol=1e-02))
except AssertionError as e:
    print('dxa values sanity check pytorch, triton:',e)

for i in range(max_l):
    try:
        torch.testing.assert_close(torch_naive_dWs[i], tri_dWs[i], rtol=0, atol=1e-02)
        print(i, 'dW i values sanity check pytorch, triton:', torch.testing.assert_close(torch_naive_dWs[i], tri_dWs[i], rtol=0, atol=1e-02))
    except AssertionError as e:
        print(i, 'dW i values sanity check pytorch, triton:',e)




try:
    torch.testing.assert_close(cpp_yi, tri_yi, rtol=0, atol=1e-02)
    print('yi values sanity check cuda, triton:', torch.testing.assert_close(cpp_yi, tri_yi, rtol=0, atol=1e-02))
except AssertionError as e:
    print('yi values sanity check cuda, triton:',e)

try:
    torch.testing.assert_close(cpp_ya, tri_ya, rtol=0, atol=1e-02)
    print('ya values sanity check cuda, triton:', torch.testing.assert_close(cpp_ya, tri_ya, rtol=0, atol=1e-02))
except AssertionError as e:
    print('ya values sanity check cuda, triton:',e)

try:
    torch.testing.assert_close(cpp_dx, tri_dx, rtol=0, atol=1e-02)
    print('dxi values sanity check cuda, triton:', torch.testing.assert_close(cpp_dx, tri_dx, rtol=0, atol=1e-02))
except AssertionError as e:
    print('dxi values sanity check cuda, triton:',e)

try:
    torch.testing.assert_close(cpp_dxa, tri_dxa, rtol=0, atol=1e-02)
    print('dxa values sanity check cuda, triton:', torch.testing.assert_close(cpp_dxa, tri_dxa, rtol=0, atol=1e-02))
except AssertionError as e:
    print('dxa values sanity check cuda, triton:',e)

for i in range(max_l):
    try:
        torch.testing.assert_close(cpp_dWs[i], tri_dWs[i], rtol=0, atol=1e-02)
        print(i, 'dW i values sanity check cuda, triton:', torch.testing.assert_close(cpp_dWs[i], tri_dWs[i], rtol=0, atol=1e-02))
    except AssertionError as e:
        print(i, 'dW i values sanity check cuda, triton:',e)



print("time pytorch-ref: {}, triton: {}, cuda: {}".format(end_time_ref-start_time_ref, end_time_triton-start_time_triton, end_time_cuda-start_time_cuda))

import pdb; pdb.set_trace()


