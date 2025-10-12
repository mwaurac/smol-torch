import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext) -> None:
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        os.makedirs(extdir, exist_ok=True)

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}',
            f'-DPython3_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE={self.build_type}',
        ]

        if sys.platform.startswith('win'):
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
        else:
            cmake_args += [f'-DCMAKE_BUILD_TYPE={self.build_type}']

        build_args = ['--config', self.build_type]

        if 'CMAKE_BUILD_PARALLEL_LEVEL' not in os.environ:
            build_args += ['--', f'-j{os.cpu_count()}']
        
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        try:
            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError("CMake build failed") from e
    @property
    def build_type(self):
        return 'Debug' if self.debug else 'Release'

setup(
    name='smol_torch',
    version='0.1.0',
    description='A lightweight Pytorch-like library',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    author='Collins Mwaura',
    author_email='left_blank@pm.me',
    url='https://github.com/mwaurac/smol-torch',
    packages=[],
    ext_modules=[CMakeExtension('smol_torch')],
    cmdclass={'build_ext': CMakeBuild},
    install_requires=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: C',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    zip_safe=False,
)
