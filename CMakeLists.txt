cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

if(DEFINED ENV{DUCC0_OPTIMIZATION})
    set (OPTFLAGS "$ENV{DUCC0_OPTIMIZATION}")
else()
    set (OPTFLAGS "native-strip")
endif()

set(PKGNAME ${SKBUILD_PROJECT_NAME})
set(PKGVERSION ${SKBUILD_PROJECT_VERSION})

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

set(CXX_SOURCES
    src/ducc0/fft/fft_inst1.cc src/ducc0/fft/fft_inst2.cc
    src/ducc0/nufft/spreadinterp_inst1.cc src/ducc0/nufft/spreadinterp_inst2.cc
    src/ducc0/healpix/healpix_base.cc src/ducc0/healpix/healpix_tables.cc
    src/ducc0/math/gl_integrator.cc src/ducc0/math/pointing.cc
    src/ducc0/math/gridding_kernel.cc src/ducc0/math/geom_utils.cc
    src/ducc0/math/wigner3j.cc src/ducc0/math/space_filling.cc
    src/ducc0/wgridder/wgridder.cc src/ducc0/wgridder/wgridder_inst1.cc
    src/ducc0/wgridder/wgridder_inst2.cc src/ducc0/wgridder/wgridder_inst3.cc
    src/ducc0/infra/string_utils.cc
    src/ducc0/infra/threading.cc src/ducc0/infra/mav.cc src/ducc0/sht/sht.cc)
set(PYMOD_SOURCES
    python/nufft_pymod.cc python/fft_pymod.cc python/sht_pymod.cc
    python/misc_pymod.cc python/totalconvolve_pymod.cc python/wgridder_pymod.cc
    python/misc_pymod.cc python/healpix_pymod.cc
    python/pointingprovider_pymod.cc python/ducc.cc)

if (DEFINED ENV{DUCC0_USE_NANOBIND})
    execute_process(
            COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE NB_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
    find_package(nanobind CONFIG REQUIRED)
    nanobind_add_module(ducc0 NOSTRIP NB_SUPPRESS_WARNINGS NOMINSIZE ${PYMOD_SOURCES} ${CXX_SOURCES})
    target_compile_definitions(ducc0 PRIVATE DUCC0_USE_NANOBIND)
else()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(ducc0 NO_EXTRAS ${PYMOD_SOURCES} ${CXX_SOURCES})
endif()

target_compile_definitions(ducc0 PRIVATE PKGNAME=${PKGNAME} PKGVERSION=${PKGVERSION})

if (NOT MSVC)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "ppc|ppc64|powerpc|powerpc64" OR (APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES "ppc|ppc64"))
        # PowerPC arch does not have -march flag.
        set(DUCC0_ARCH_FLAGS "-mtune=native" CACHE STRING "Compiler flags for specifying target architecture.")
    else ()
        set(DUCC0_ARCH_FLAGS "-march=native" CACHE STRING "Compiler flags for specifying target architecture.")
    endif ()
    message(STATUS "Using GCC/Clang flags: ${DUCC0_ARCH_FLAGS}")
else ()
    # Check for AVX, AVX512 and SSE support
    message(STATUS "Checking for AVX, AVX512 and SSE support")
    try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
            ${CMAKE_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckAVX.cpp
            COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT
            RUN_OUTPUT_VARIABLE RUN_OUTPUT)
    if (RUN_OUTPUT MATCHES "AVX512")
        set(DUCC0_ARCH_FLAGS "/arch:AVX512" CACHE STRING "Compiler flags for specifying target architecture.")
    elseif (RUN_OUTPUT MATCHES "AVX")
        set(DUCC0_ARCH_FLAGS "/arch:AVX" CACHE STRING "Compiler flags for specifying target architecture.")
    elseif (RUN_OUTPUT MATCHES "SSE")
        set(DUCC0_ARCH_FLAGS "/arch:SSE" CACHE STRING "Compiler flags for specifying target architecture.")
    else ()
        set(DUCC0_ARCH_FLAGS "" CACHE STRING "Compiler flags for specifying target architecture.")
    endif ()
    message(STATUS "CPU supports: ${RUN_OUTPUT}")
    message(STATUS "Using MSVC flags: ${DUCC0_ARCH_FLAGS}")
endif ()

include_directories(${CMAKE_SOURCE_DIR}/src)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    SET(DUCC0_WARN_FLAGS
        -Wfatal-errors -Wfloat-conversion -W -Wall -Wstrict-aliasing
        -Wwrite-strings -Wredundant-decls -Woverloaded-virtual -Wcast-qual
        -Wcast-align -Wpointer-arith -Wnon-virtual-dtor
        -Wzero-as-null-pointer-constant)
#    set (DUCC0_OPT_FLAGS -O3 -ffp-contract=fast -fexcess-precision=fast -fno-math-errno -fno-signed-zeros -fno-trapping-math -fassociative-math -freciprocal-math)
    # we can use -ffast-math here because the flags will not be used for linking
    set (DUCC0_OPT_FLAGS -O3 -ffast-math)
    set (DUCC0_NOOPT_FLAGS -O0)
    set (DUCC0_DEBUG_FLAGS -g)
    set (DUCC0_STRIP_FLAGS -s)
endif()
target_compile_options(ducc0 PRIVATE ${DUCC0_WARN_FLAGS})

if (OPTFLAGS MATCHES "^native")
  target_compile_options(ducc0 PRIVATE ${DUCC0_ARCH_FLAGS})
endif()
if (OPTFLAGS MATCHES "-debug$")
  target_compile_options(ducc0 PRIVATE ${DUCC0_DEBUG_FLAGS})
elseif (OPTFLAGS MATCHES "-strip$")
  target_link_options(ducc0 PRIVATE ${DUCC0_STRIP_FLAGS})
endif()
if (OPTFLAGS MATCHES "^none")
  target_compile_options(ducc0 PRIVATE ${DUCC0_NOOPT_FLAGS})
else()
  target_compile_options(ducc0 PRIVATE ${DUCC0_OPT_FLAGS})
endif()

find_package(Threads REQUIRED)
target_link_libraries(ducc0 PRIVATE Threads::Threads)

target_compile_features(ducc0 PRIVATE cxx_std_17)

set_property(TARGET ducc0 PROPERTY INTERPROCEDURAL_OPTIMIZATION True)

install(TARGETS ducc0 LIBRARY DESTINATION .)
