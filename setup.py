from setuptools import setup, Extension
import sys

pkgname = 'cxxbase1'

class _deferred_pybind11_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


include_dirs = ['./src/',
                _deferred_pybind11_include(True),
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
    python_module_link_args += ['-march=native', '-Wl,-rpath,$ORIGIN' ,'-s']

# if you don't want debugging info, add "-s" to python_module_link_args

def get_extension_modules():
    return [Extension(pkgname+'.pypocketfft',
                      language='c++',
                      sources=['pypocketfft/pypocketfft.cc',
                               'src/mr_util/infra/threading.cc'],
                      depends=['src/mr_util/infra/useful_macros.h',
                               'src/mr_util/math/fft.h',
                               'src/mr_util/math/fft1d.h',
                               'src/mr_util/infra/mav.h',
                               'src/mr_util/infra/threading.h',
                               'src/mr_util/infra/aligned_array.h',
                               'src/mr_util/infra/simd.h',
                               'src/mr_util/math/cmplx.h',
                               'src/mr_util/math/unity_roots.h',
                               'src/mr_util/infra/error_handling.h',
                               'src/mr_util/bindings/pybind_utils.h',
                               'setup.py'],
                      include_dirs=include_dirs,
                      define_macros=define_macros,
                      extra_compile_args=extra_compile_args,
                      extra_link_args=python_module_link_args),
            Extension(pkgname+'.pysharp',
                      language='c++',
                      sources=['pysharp/pysharp.cc',
                               'src/mr_util/infra/threading.cc',
                               'src/mr_util/infra/string_utils.cc',
                               'src/mr_util/sharp/sharp.cc',
                               'src/mr_util/sharp/sharp_core.cc',
                               'src/mr_util/sharp/sharp_geomhelpers.cc',
                               'src/mr_util/sharp/sharp_almhelpers.cc',
                               'src/mr_util/sharp/sharp_ylmgen.cc'],
                      depends=['src/mr_util/fft1d.h',
                               'src/mr_util/fft.h',
                               'src/mr_util/infra/threading.h',
                               'src/mr_util/infra/mav.h',
                               'src/mr_util/math_utils.h',
                               'src/mr_util/infra/aligned_array.h',
                               'src/mr_util/math/gl_integrator.h',
                               'src/mr_util/infra/simd.h',
                               'src/mr_util/math/cmplx.h',
                               'src/mr_util/infra/string_utils.h',
                               'src/mr_util/infra/timers.h',
                               'src/mr_util/math/constants.h',
                               'src/mr_util/math/unity_roots.h',
                               'src/mr_util/infra/error_handling.h',
                               'src/mr_util/infra/useful_macros.h',
                               'src/mr_util/bindings/pybind_utils.h',
                               'src/mr_util/sharp/sharp.h',
                               'src/mr_util/sharp/sharp_internal.h',
                               'src/mr_util/sharp/sharp_geomhelpers.h',
                               'src/mr_util/sharp/sharp_almhelpers.h'
                               'setup.py'],
                      include_dirs=include_dirs,
                      define_macros=define_macros,
                      extra_compile_args=extra_compile_args,
                      extra_link_args=python_module_link_args),
            Extension(pkgname+'.pyHealpix',
                      language='c++',
                      sources=['pyHealpix/pyHealpix.cc',
                               'src/mr_util/math/geom_utils.cc',
                               'src/mr_util/math/pointing.cc',
                               'src/mr_util/infra/string_utils.cc',
                               'src/mr_util/math/space_filling.cc',
                               'src/mr_util/healpix/healpix_base.cc',
                               'src/mr_util/healpix/healpix_tables.cc'],
                      depends=['src/mr_util/infra/mav.h',
                               'src/mr_util/math/math_utils.h',
                               'src/mr_util/math/space_filling.h',
                               'src/mr_util/math/rangeset.h',
                               'src/mr_util/infra/string_utils.h',
                               'src/mr_util/math/geom_utils.h',
                               'src/mr_util/math/pointing.h',
                               'src/mr_util/math/vec3.h',
                               'src/mr_util/math/constants.h',
                               'src/mr_util/infra/error_handling.h',
                               'src/mr_util/healpix/healpix_base.h',
                               'src/mr_util/healpix/healpix_tables.h',
                               'src/mr_util/bindings/pybind_utils.h',
                               'setup.py'],
                      include_dirs=include_dirs,
                      define_macros=define_macros,
                      extra_compile_args=extra_compile_args,
                      extra_link_args=python_module_link_args),
            Extension(pkgname+'.nifty_gridder',
                      sources=['nifty_gridder/nifty_gridder.cc',
                               'src/mr_util/infra/threading.cc'],
                      depends=['src/mr_util/infra/useful_macros.h',
                               'src/mr_util/math/constants.h',
                               'src/mr_util/math/gl_integrator.h',
                               'src/mr_util/infra/error_handling.h',
                               'src/mr_util/math/fft.h',
                               'src/mr_util/math/fft1d.h',
                               'src/mr_util/infra/threading.h',
                               'src/mr_util/infra/aligned_array.h',
                               'src/mr_util/infra/simd.h',
                               'src/mr_util/infra/mav.h',
                               'src/mr_util/math/cmplx.h',
                               'src/mr_util/math/unity_roots.h',
                               'src/mr_util/math/es_kernel.h',
                               'src/mr_util/bindings/pybind_utils.h',
                               'nifty_gridder/gridder_cxx.h',
                               'setup.py'],
                      include_dirs=include_dirs + ['./nifty_gridder'],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=python_module_link_args),
           Extension(pkgname+'.pyinterpol_ng',
                      language='c++',
                      sources=['pyinterpol_ng/pyinterpol_ng.cc',
                               'src/mr_util/infra/threading.cc',
                               'src/mr_util/sharp/sharp.cc',
                               'src/mr_util/sharp/sharp_core.cc',
                               'src/mr_util/sharp/sharp_geomhelpers.cc',
                               'src/mr_util/sharp/sharp_almhelpers.cc',
                               'src/mr_util/sharp/sharp_ylmgen.cc'],
                      depends=[
                               'src/mr_util/math/fft1d.h',
                               'src/mr_util/math/fft.h',
                               'src/mr_util/infra/threading.h',
                               'src/mr_util/infra/mav.h',
                               'src/mr_util/math/math_utils.h',
                               'src/mr_util/infra/aligned_array.h',
                               'src/mr_util/math/gl_integrator.h',
                               'src/mr_util/infra/simd.h',
                               'src/mr_util/math/cmplx.h',
                               'src/mr_util/infra/string_utils.h',
                               'src/mr_util/infra/timers.h',
                               'src/mr_util/math/constants.h',
                               'src/mr_util/math/unity_roots.h',
                               'src/mr_util/math/es_kernel.h',
                               'src/mr_util/infra/error_handling.h',
                               'src/mr_util/infra/useful_macros.h',
                               'src/mr_util/bindings/pybind_utils.h',
                               'src/mr_util/sharp/sharp.h',
                               'src/mr_util/sharp/sharp_internal.h',
                               'src/mr_util/sharp/sharp_geomhelpers.h',
                               'src/mr_util/sharp/sharp_almhelpers.h',
                               'setup.py',
                               'pyinterpol_ng/interpol_ng.h',
                               'pyinterpol_ng/alm.h'],
                      include_dirs=include_dirs + ['./pyinterpol_ng'],
                      define_macros=define_macros,
                      extra_compile_args=extra_compile_args,
                      extra_link_args=python_module_link_args),
                ]


setup(name=pkgname,
      version='0.0.1',
      description='Various neat modules',
      include_package_data=True,
      author='Martin Reinecke',
      author_email='martin@mpa-garching.mpg.de',
      packages=[],
      setup_requires=['numpy>=1.15.0', 'pybind11>=2.2.4'],
      ext_modules=get_extension_modules(),
      install_requires=['numpy>=1.15.0', 'pybind11>=2.2.4']
      )
