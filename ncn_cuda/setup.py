# https://brsoff.github.io/tutorials/advanced/cpp_extension.html
# https://medium.com/@justygwen/teach-you-to-implement-pytorch-cuda-operators-like-teaching-a-loved-one-dbd572410558
# https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ncn_cuda',
    ext_modules=[
        CUDAExtension('ncn_cuda_module', [
            'ncn_cuda/ncn_cuda.cpp',
            'ncn_cuda/ncn_fwd_cuda_kernel.cu',
            'ncn_cuda/ncn_bwd_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })