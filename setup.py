from setuptools import setup, Extension

smol_torch_extension = Extension(
    'smol_torch',
    sources=['src/smol_torch.c', 'src/tensor.c', 'library.c'],
    include_dirs=['include'],
)

setup(
    name='smol_torch',
    version='0.1',
    description='A small torch-like library',
    ext_modules=[smol_torch_extension],
)
