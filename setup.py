import sys
import os.path
import itertools
from glob import iglob
import os

from setuptools import setup, Extension
import pybind11

pkgname = 'ducc0'
version = '0.30.0'

tmp = os.getenv("DUCC0_CFLAGS", "").split(" ")
user_cflags = [x for x in tmp if x != ""]
tmp = os.getenv("DUCC0_LFLAGS", "").split(" ")
user_lflags = [x for x in tmp if x != ""]
tmp = os.getenv("DUCC0_FLAGS", "").split(" ")
tmp = [x for x in tmp if x != ""]
user_cflags += tmp
user_lflags += tmp

compilation_strategy = os.getenv('DUCC0_OPTIMIZATION', 'native-strip')
if compilation_strategy not in ['none', 'none-debug', 'none-strip', 'portable', 'portable-debug', 'portable-strip', 'native', 'native-debug', 'native-strip']:
    raise RuntimeError('unknown compilation strategy')
do_debug = compilation_strategy in ['none-debug', 'portable-debug', 'native-debug']
do_strip = compilation_strategy in ['none-strip', 'portable-strip', 'native-strip']
do_optimize = compilation_strategy not in ['none', 'none-debug', 'none-strip']
do_native = compilation_strategy in ['native', 'native-debug', 'native-strip']

def _print_env():
    import platform
    print("")
    print("Build environment:")
    print("Platform:     ", platform.platform())
    print("Machine:      ", platform.machine())
    print("System:       ", platform.system())
    print("Architecture: ", platform.architecture())
    print("")

def _get_files_by_suffix(directory, suffix):
    path = directory
    iterable_sources = (iglob(os.path.join(root, '*.'+suffix))
                        for root, dirs, files in os.walk(path))
    return list(itertools.chain.from_iterable(iterable_sources))


include_dirs = ['.', './src/',
                pybind11.get_include(True),
                pybind11.get_include(False)]

extra_compile_args = ['-std=c++17', '-fvisibility=hidden']

if do_debug:
    extra_compile_args += ['-g']
else:
    extra_compile_args += ['-g0']

if do_optimize:
    extra_compile_args += ['-ffast-math', '-O3']
else:
    extra_compile_args += ['-O0']

if do_native:
    extra_compile_args += ['-march=native']

python_module_link_args = []

define_macros = [("PKGNAME", pkgname),
                 ("PKGVERSION", version)]

if sys.platform == 'darwin':
    extra_compile_args += ['-mmacosx-version-min=10.14', '-pthread']
    python_module_link_args += ['-mmacosx-version-min=10.14', '-pthread']
elif sys.platform == 'win32':
    extra_compile_args = ['/EHsc', '/std:c++17']
    if do_optimize:
        extra_compile_args += ['/Ox']
else:
    if do_optimize:
        extra_compile_args += ['-Wfatal-errors',
                               '-Wfloat-conversion',
                               '-W',
                               '-Wall',
                               '-Wstrict-aliasing',
                               '-Wwrite-strings',
                               '-Wredundant-decls',
                               '-Woverloaded-virtual',
                               '-Wcast-qual',
                               '-Wcast-align',
                               '-Wpointer-arith',
                               '-Wnon-virtual-dtor',
                               '-Wzero-as-null-pointer-constant']
    extra_compile_args += ['-pthread']
    python_module_link_args += ['-Wl,-rpath,$ORIGIN', '-pthread']
    if do_native:
        python_module_link_args += ['-march=native']
    if do_strip:
        python_module_link_args += ['-s']

extra_compile_args += user_cflags
python_module_link_args += user_lflags

depfiles = (_get_files_by_suffix('.', 'h') +
            _get_files_by_suffix('.', 'cc') +
            ['setup.py'])

extensions = [Extension(pkgname,
                        language='c++',
                        sources=['python/ducc.cc'],
                        depends=depfiles,
                        include_dirs=include_dirs,
                        define_macros=define_macros,
                        extra_compile_args=extra_compile_args,
                        extra_link_args=python_module_link_args)]

_print_env()

setup(name=pkgname,
      version=version,
      ext_modules = extensions
      )
