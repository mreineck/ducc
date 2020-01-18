from setuptools import setup, Extension
import sys


class _deferred_pybind11_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


include_dirs = ['../', './', _deferred_pybind11_include(True),
                _deferred_pybind11_include()]
extra_compile_args = ['-Wall', '-Wextra', '-Wfatal-errors', '-Wstrict-aliasing=2', '-Wwrite-strings', '-Wredundant-decls', '-Woverloaded-virtual', '-Wcast-qual', '-Wcast-align', '-Wpointer-arith', '-Wfloat-conversion']
#, '-Wsign-conversion', '-Wconversion'
python_module_link_args = []

if sys.platform == 'darwin':
    import distutils.sysconfig
    extra_compile_args += ['--std=c++11', '--stdlib=libc++', '-mmacosx-version-min=10.9']
    vars = distutils.sysconfig.get_config_vars()
    vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '')
    python_module_link_args+=['-bundle']
else:
    extra_compile_args += ['--std=c++11', '-march=native', '-O3', '-ffast-math']
    python_module_link_args += ['-march=native', '-Wl,-rpath,$ORIGIN', '-ffast-math']

# if you don't want debugging info, add "-s" to python_module_link_args


def get_extension_modules():
    return [Extension('nifty_gridder',
                      sources=['nifty_gridder.cc', '../mr_util/threading.cc', '../mr_util/error_handling.cc'],
                      depends=['../mr_util/error_handling.h', '../mr_util/fft.h', '../mr_util/threading.h',
                               '../mr_util/aligned_array.h', '../mr_util/simd.h', '../mr_util/mav.h',
                               '../mr_util/cmplx.h', '../mr_util/unity_roots.h',
                               'setup.py', 'gridder_cxx.h'],
                      include_dirs=include_dirs,
                      extra_compile_args=extra_compile_args,
                      extra_link_args=python_module_link_args)]

setup(name='nifty_gridder',
      version='0.0.1',
      description='Gridding/Degridding helper library for NIFTy',
      include_package_data=True,
      author='Martin Reinecke',
      author_email='martin@mpa-garching.mpg.de',
      packages=[],
      setup_requires=['numpy>=1.15.0', 'pybind11>=2.2.4'],
      ext_modules=get_extension_modules(),
      install_requires=['numpy>=1.15.0', 'pybind11>=2.2.4']
      )
