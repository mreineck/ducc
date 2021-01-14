import sys
import os.path
import itertools
from glob import iglob
import os

from setuptools import setup, Extension
import pybind11

pkgname = 'ducc0'
version = '0.8.0'

user_cflags = os.getenv("DUCC0_CFLAGS", "").split(" ")
user_cflags = [x for x in user_cflags if x != ""]
user_lflags = os.getenv("DUCC0_LFLAGS", "").split(" ")
user_lflags = [x for x in user_lflags if x != ""]

compilation_strategy = os.getenv('DUCC0_OPTIMIZATION', 'native')
if compilation_strategy not in ['none', 'portable', 'portable-debug', 'native', 'native-debug']:
    raise RuntimeError('unknown compilation strategy')
do_debug = compilation_strategy in ['portable-debug', 'native-debug']
do_optimize = compilation_strategy not in ['none']
do_native = compilation_strategy in ['native', 'native-debug']

def _get_files_by_suffix(directory, suffix):
    path = directory
    iterable_sources = (iglob(os.path.join(root, '*.'+suffix))
                        for root, dirs, files in os.walk(path))
    return list(itertools.chain.from_iterable(iterable_sources))


include_dirs = ['.', './src/',
                pybind11.get_include(True),
                pybind11.get_include(False)]

extra_compile_args = ['-std=c++17']

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
                 ("PKGVERSION", '"%s"' % version)]

if sys.platform == 'darwin':
    import distutils.sysconfig
    extra_compile_args += ['-mmacosx-version-min=10.14']
    python_module_link_args += ['-mmacosx-version-min=10.14', '-bundle']
    cfg_vars = distutils.sysconfig.get_config_vars()
    cfg_vars['LDSHARED'] = cfg_vars['LDSHARED'].replace('-bundle', '')
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
                               '-Wstrict-aliasing=2',
                               '-Wwrite-strings',
                               '-Wredundant-decls',
                               '-Woverloaded-virtual',
                               '-Wcast-qual',
                               '-Wcast-align',
                               '-Wpointer-arith']

    python_module_link_args += ['-Wl,-rpath,$ORIGIN']
    if do_native:
        python_module_link_args += ['-march=native']

extra_compile_args += user_cflags
python_module_link_args += user_lflags

# if you want debugging info, remove the "-s" from python_module_link_args
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


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(name=pkgname,
      version=version,
      description='Distinctly useful code collection',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://gitlab.mpcdf.mpg.de/mtr/ducc',
      include_package_data=True,
      author='Martin Reinecke',
      author_email='martin@mpa-garching.mpg.de',
      packages=[],
      python_requires=">=3.6",
      ext_modules=extensions,
      install_requires=['numpy>=1.17.0'],
      license="GPLv2",
      )
