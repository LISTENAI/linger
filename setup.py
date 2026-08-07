import os
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


sources = ['linger/kernel/cpu/extension.cpp']
cpu_only = os.getenv("LINGER_CPU_ONLY", "0") == "1"

torch_version = torch.__version__
if '+' in torch_version:
    torch_version = torch_version.split('+')[0]
versions = torch_version.split('.')
version_maj = int(versions[0])
version_min = int(versions[1])
version_patch = int(versions[2])
# version 1.5.1
if version_maj*100+version_min*10 + version_patch >= 151:
    sources.append('linger/kernel/cpu/util_kernel.cpp')
    sources.append('linger/kernel/cpu/venus_qsoftmax_kernel.cpp')
    sources.append('linger/kernel/cpu/venusa_qsoftmax_kernel.cpp')
    sources.append('linger/kernel/cpu/arcs_qsoftmax_kernel.cpp')
    sources.append('linger/kernel/cpu/venus_qsigmoid_kernel.cpp')
    sources.append('linger/kernel/cpu/venusa_qsigmoid_kernel.cpp')
    sources.append('linger/kernel/cpu/arcs_qsigmoid_kernel.cpp')
    sources.append('linger/kernel/cpu/venus_qtanh_kernel.cpp')
    sources.append('linger/kernel/cpu/venusa_qtanh_kernel.cpp')
    sources.append('linger/kernel/cpu/arcs_qtanh_kernel.cpp')
    sources.append('linger/kernel/cpu/qlayernorm_kernel.cpp')
    if not cpu_only:
        sources.append('linger/kernel/gpu/util_kernel.cu')
        sources.append('linger/kernel/gpu/venus_qsoftmax_kernel.cu')
        sources.append('linger/kernel/gpu/venusa_qsoftmax_kernel.cu')
        sources.append('linger/kernel/gpu/arcs_qsoftmax_kernel.cu')
        sources.append('linger/kernel/gpu/venus_qsigmoid_kernel.cu')
        sources.append('linger/kernel/gpu/venusa_qsigmoid_kernel.cu')
        sources.append('linger/kernel/gpu/arcs_qsigmoid_kernel.cu')
        sources.append('linger/kernel/gpu/venus_qtanh_kernel.cu')
        sources.append('linger/kernel/gpu/venusa_qtanh_kernel.cu')
        sources.append('linger/kernel/gpu/arcs_qtanh_kernel.cu')
        sources.append('linger/kernel/gpu/qlayernorm_kernel.cu')
        sources.append('linger/kernel/gpu/fake_quant_kernel.cu')

extension_cls = CppExtension if cpu_only else CUDAExtension
extra_compile_args = {'cxx': ['-g', '-O2', '-Wall', '-Wextra', '-Wno-unused-parameter', '-Wno-missing-field-initializers', '-fPIC', '-fopenmp']}
if cpu_only:
    extra_compile_args['cxx'].append('-DLINGER_CPU_ONLY')
else:
    extra_compile_args['nvcc'] = ['-O2',
                                  '--use_fast_math',
                                  '--ftz=true',
                                  '-Xcompiler', '-fPIC',
                                  '-Xcompiler', '-fopenmp',
                                  '--compiler-options', '-Wall',
                                  '--compiler-options', '-Wextra']

setup(
    name="linger",
    version="3.1.0",
    description="linger is package of fix training",
    author="ListenAI",
    ext_modules=[
        extension_cls('lingerext',
                      sources=sources,
                      extra_compile_args=extra_compile_args),
    ],


    cmdclass={
        'build_ext': BuildExtension
    },
    packages=find_packages(exclude=['tools', 'tools.*'])+ ['linger.checker'],
    package_dir={
        'linger.checker': 'tools/checker'
    },
    include_package_data=True,
    classifiers=[
        "Operating System :: OS Independent",
        "Intended Audience :: Developers and Researchers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: python",
    ],
)
