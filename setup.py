import sys
import os.path
import itertools
from glob import iglob

from setuptools import setup, Extension
import pybind11

pkgname = 'ducc0'
version = '0.3.1'


def _get_files_by_suffix(directory, suffix):
    path = directory
    iterable_sources = (iglob(os.path.join(root, '*.'+suffix))
                        for root, dirs, files in os.walk(path))
    return list(itertools.chain.from_iterable(iterable_sources))


include_dirs = ['.', './src/',
                pybind11.get_include(True),
                pybind11.get_include(False)]

extra_compile_args = ['--std=c++17', '-march=native', '-ffast-math', '-O3']

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
    extra_compile_args = ['/Ox', '/EHsc', '/std:c++17']
else:
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

    python_module_link_args += ['-march=native',
                                '-Wl,-rpath,$ORIGIN',
                                '-s']

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
