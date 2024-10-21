from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_conv2d',
    ext_modules=[
        CUDAExtension('custom_conv2d', [
            'custom_conv2d_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
