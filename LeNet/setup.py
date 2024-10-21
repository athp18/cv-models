from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='conv2d',
    ext_modules=[
        CUDAExtension('conv2d', [
            'conv2d_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
