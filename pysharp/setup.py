from setuptools import setup, Extension
import sys


class _deferred_pybind11_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


include_dirs = ['..', _deferred_pybind11_include(True),
                _deferred_pybind11_include()]
extra_compile_args = ['--std=c++17', '-march=native', '-ffast-math', '-O3']
python_module_link_args = []
define_macros = []

if sys.platform == 'darwin':
    import distutils.sysconfig
    extra_compile_args += ['-mmacosx-version-min=10.9']
    python_module_link_args += ['-mmacosx-version-min=10.9', '-bundle']
    vars = distutils.sysconfig.get_config_vars()
    vars['LDSHARED'] = vars['LDSHARED'].replace('-bundle', '')
elif sys.platform == 'win32':
    extra_compile_args = ['/Ox', '/EHsc', '/std:c++17']
else:
    extra_compile_args += ['-Wfatal-errors', '-Wfloat-conversion', '-W', '-Wall', '-Wstrict-aliasing=2', '-Wwrite-strings', '-Wredundant-decls', '-Woverloaded-virtual', '-Wcast-qual', '-Wcast-align', '-Wpointer-arith']
    python_module_link_args += ['-march=native', '-ffast-math', '-Wl,-rpath,$ORIGIN']

# if you don't want debugging info, add "-s" to python_module_link_args

def get_extension_modules():
    return [Extension('pysharp',
                      language='c++',
                      sources=['pysharp.cc','../mr_util/threading.cc','../mr_util/string_utils.cc',
                               '../libsharp2/sharp.cc', '../libsharp2/sharp_core.cc', '../libsharp2/sharp_geomhelpers.cc',
                               '../libsharp2/sharp_almhelpers.cc','../libsharp2/sharp_ylmgen.cc'],
                      depends=['../mr_util/string_utils.h', '../mr_util/fft.h', '../mr_util/mav.h', '../mr_util/threading.h',
                               '../mr_util/aligned_array.h', '../mr_util/simd.h',
                               '../mr_util/cmplx.h', '../mr_util/unity_roots.h', '../mr_util/error_handling.h',
                               'setup.py'],

                      include_dirs=include_dirs,
                      define_macros=define_macros,
                      extra_compile_args=extra_compile_args,
                      extra_link_args=python_module_link_args)]


setup(name='pysharp',
      version='0.0.1',
      description='Python bindings for some libsharp functionality',
      include_package_data=True,
      author='Martin Reinecke',
      author_email='martin@mpa-garching.mpg.de',
      packages=[],
      setup_requires=['numpy>=1.15.0', 'pybind11>=2.2.4'],
      ext_modules=get_extension_modules(),
      install_requires=['numpy>=1.15.0', 'pybind11>=2.2.4']
      )
