import torch
from ncn_triton.ncn_triton import NCNNetTestTriton
from torch_ref.ncn_pytorch_invertible import RefInvTorchNaive
from torch_ref.ncn_pytorch import RefTorchNaive
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



batch_size = 4
n_head = 4
seq_len = 256
embd = 128

alpha = 0.9
n_cache = 32
activation = "tanh"

max_l = int(math.log2(seq_len // n_cache))
max_l = 3
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


# Ref
start_time_ref = time.time()

ref_net = RefTorchNaive(alpha, activation, max_l, n_cache, n_head, Ws)#.cuda()

ref_yi, ref_ya = ref_net(x, xa)

ref_yi.backward(dyi)
#ref_ya.backward(dya)

ref_dWs = []

for i in range(0, max_l):
    ref_dW = ref_net.weights[i].grad.clone()
    ref_net.weights[i].grad = None
    ref_dWs.append(ref_dW)


ref_dx, x.grad = x.grad.clone(), None
ref_dxa, xa.grad = xa.grad.clone(), None

end_time_ref = time.time()


# Ref invertible, with reconstruction error
start_time_ref_invertible = time.time()

ref_inv_net = RefInvTorchNaive(alpha, activation, max_l, n_cache, n_head, Ws)#.cuda()

ref_inv_yi, ref_inv_ya = ref_inv_net(x, xa)

ref_inv_yi.backward(dyi)
#ref_inv_ya.backward(dya)

ref_inv_dWs = []

for i in range(0, max_l):
    ref_inv_dW = ref_inv_net.weights[i].grad.clone()
    ref_inv_net.weights[i].grad = None
    ref_inv_dWs.append(ref_inv_dW)


ref_inv_dx, x.grad = x.grad.clone(), None
ref_inv_dxa, xa.grad = xa.grad.clone(), None

end_time_ref_invertible = time.time()



# NCN Cuda version

start_time_cuda = time.time()

ncn_net_cuda = NCNNetTestCuda(alpha, activation, n_cache, n_head, max_l, Ws)#.cuda()
cpp_yi, cpp_ya = ncn_net_cuda(x, xa)

cpp_yi.backward(dyi)
#cpp_ya.backward(dya)

cpp_dWs = []

for i in range(max_l):
    cpp_dW = ncn_net_cuda.weights[i].grad.clone()
    ncn_net_cuda.weights[i].grad = None
    cpp_dWs.append(cpp_dW)

cpp_dx, x.grad = x.grad.clone(), None
cpp_dxa, xa.grad = xa.grad.clone(), None

end_time_cuda = time.time()

# NCN Triton version

start_time_triton = time.time()

ncn_net_triton = NCNNetTestTriton(alpha, activation, n_cache, n_head, max_l, Ws)#.cuda()
tri_yi, tri_ya = ncn_net_triton(x, xa)

tri_yi.backward(dyi)
#tri_ya.backward(dya)

tri_dWs = []

for i in range(max_l):
    tri_dW = ncn_net_triton.weights[i].grad.clone()
    ncn_net_triton.weights[i].grad = None
    tri_dWs.append(tri_dW)

tri_dx, x.grad = x.grad.clone(), None
tri_dxa, xa.grad = xa.grad.clone(), None

end_time_triton = time.time()


print("Testing reference methods")
try:
    torch.testing.assert_close(ref_yi, ref_inv_yi, rtol=0, atol=1e-02)
    print('yi values sanity check ref vs invertible ref:', torch.testing.assert_close(ref_yi, ref_inv_yi, rtol=0, atol=1e-02))
except AssertionError as e:
    print("yi values sanity check ref vs invertible ref:", e)

try:
    torch.testing.assert_close(ref_ya, ref_inv_ya, rtol=0, atol=1e-02)
    print('ya values sanity check ref vs invertible ref:', torch.testing.assert_close(ref_ya, ref_inv_ya, rtol=0, atol=1e-02))
except AssertionError as e:
    print('ya values sanity check ref vs invertible ref:' ,e)

try:
    torch.testing.assert_close(ref_dx, ref_inv_dx, rtol=0, atol=1e-02)
    print('dxi values sanity check ref vs invertible ref:', torch.testing.assert_close(ref_dx, ref_inv_dx, rtol=0, atol=1e-02))
except AssertionError as e:
    print('dxi values sanity check ref vs invertible ref:', e)

try:
    torch.testing.assert_close(ref_dxa, ref_inv_dxa, rtol=0, atol=1e-02)
    print('dxa values sanity check ref vs invertible ref:', torch.testing.assert_close(ref_dxa, ref_inv_dxa, rtol=0, atol=1e-02))
except AssertionError as e:
    print('dxa values sanity check ref vs invertible ref:',e)

for i in range(max_l):
    try:
        torch.testing.assert_close(ref_dWs[i], ref_inv_dWs[i], rtol=0, atol=1e-02)
        print(i ,'dW  values sanity check ref vs invertible ref:', torch.testing.assert_close(ref_dWs[i], ref_inv_dWs[i], rtol=0, atol=1e-02))
    except AssertionError as e:
        print(i, 'dW values sanity check ref vs invertible ref:',e)



print("Testing reference vs triton")

try:
    torch.testing.assert_close(ref_yi, tri_yi, rtol=0, atol=1e-02)
    print('yi values sanity check ref vs triton:', torch.testing.assert_close(ref_yi, tri_yi, rtol=0, atol=1e-02))
except AssertionError as e:
    print('yi values sanity check ref vs triton:',e)

try:
    torch.testing.assert_close(ref_ya, tri_ya, rtol=0, atol=1e-02)
    print('ya values sanity checkref vs triton:', torch.testing.assert_close(ref_ya, tri_ya, rtol=0, atol=1e-02))
except AssertionError as e:
    print('ya values sanity check ref vs triton:',e)

try:
    torch.testing.assert_close(ref_dx, tri_dx, rtol=0, atol=1e-02)
    print('dxi values sanity check ref vs triton:', torch.testing.assert_close(ref_dx, tri_dx, rtol=0, atol=1e-02))
except AssertionError as e:
    print('dxi values sanity check ref vs triton:',e)

try:
    torch.testing.assert_close(ref_dxa, tri_dxa, rtol=0, atol=1e-02)
    print('dxa values sanity check ref vs triton:', torch.testing.assert_close(ref_dxa, tri_dxa, rtol=0, atol=1e-02))
except AssertionError as e:
    print('dxa values sanity check ref vs triton:',e)

for i in range(max_l):
    try:
        torch.testing.assert_close(ref_dWs[i], tri_dWs[i], rtol=0, atol=1e-02)
        print(i, 'dW i values sanity check ref vs triton:', torch.testing.assert_close(ref_dWs[i], tri_dWs[i], rtol=0, atol=1e-02))
    except AssertionError as e:
        print(i, 'dW i values sanity check ref vs triton:',e)


print("Testing reference vs cuda")



try:
    torch.testing.assert_close(ref_yi, cpp_yi, rtol=0, atol=1e-02)
    print('yi values sanity check ref vs cuda:', torch.testing.assert_close(ref_yi, cpp_yi, rtol=0, atol=1e-02))
except AssertionError as e:
    print("yi values sanity check ref vs cuda:", e)

try:
    torch.testing.assert_close(ref_ya, cpp_ya, rtol=0, atol=1e-02)
    print('ya values sanity check ref vs cuda:', torch.testing.assert_close(ref_ya, cpp_ya, rtol=0, atol=1e-02))
except AssertionError as e:
    print('ya values sanity check ref vs cuda:' ,e)

try:
    torch.testing.assert_close(ref_dx, cpp_dx, rtol=0, atol=1e-02)
    print('dxi values sanity check ref vs cuda:', torch.testing.assert_close(ref_dx, cpp_dx, rtol=0, atol=1e-02))
except AssertionError as e:
    print('dxi values sanity check ref vs cuda:', e)

try:
    torch.testing.assert_close(ref_dxa, cpp_dxa, rtol=0, atol=1e-02)
    print('dxa values sanity check ref vs cuda:', torch.testing.assert_close(ref_dxa, cpp_dxa, rtol=0, atol=1e-02))
except AssertionError as e:
    print('dxa values sanity check ref vs cuda:',e)

for i in range(max_l):
    try:
        torch.testing.assert_close(ref_dWs[i], cpp_dWs[i], rtol=0, atol=1e-02)
        print(i ,'dW  values sanity check ref vs cuda:', torch.testing.assert_close(ref_dWs[i], cpp_dWs[i], rtol=0, atol=1e-02))
    except AssertionError as e:
        print(i, 'dW values sanity checkref vs cuda:',e)



print("Testing cuda vs triton")

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



print("times ref: {}, ref_invertible: {}, triton: {}, cuda: {}".format(end_time_ref-start_time_ref, end_time_ref_invertible-start_time_ref_invertible,end_time_triton-start_time_triton, end_time_cuda-start_time_cuda))


