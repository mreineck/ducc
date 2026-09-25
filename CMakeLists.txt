cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION True)

if(DEFINED ENV{DUCC0_OPTIMIZATION})
    set (OPTFLAGS "$ENV{DUCC0_OPTIMIZATION}")
else()
    set (OPTFLAGS "native-strip")
endif()
if (DEFINED ENV{DUCC0_MULTIARCH} AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set (MULTIARCH ON)
else()
    set (MULTIARCH OFF)
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
    src/ducc0/wgridder/wgridder_inst1.cc
    src/ducc0/wgridder/wgridder_inst2.cc src/ducc0/wgridder/wgridder_inst3.cc
    src/ducc0/infra/string_utils.cc
    src/ducc0/infra/threading.cc src/ducc0/infra/mav.cc src/ducc0/sht/sht.cc)
set(PYMOD_SOURCES
    python/nufft_pymod.cc python/fft_pymod.cc python/sht_pymod.cc
    python/misc_pymod.cc python/totalconvolve_pymod.cc python/wgridder_pymod.cc
    python/misc_pymod.cc python/healpix_pymod.cc
    python/pointingprovider_pymod.cc python/ducc.cc)

if (MULTIARCH)
    add_library(${PKGNAME}_lib_v1 OBJECT ${PYMOD_SOURCES} ${CXX_SOURCES})
    add_library(${PKGNAME}_lib_v3 OBJECT ${PYMOD_SOURCES} ${CXX_SOURCES})
    add_library(${PKGNAME}_lib_v4 OBJECT ${PYMOD_SOURCES} ${CXX_SOURCES})
else()
    add_library(${PKGNAME}_lib OBJECT ${PYMOD_SOURCES} ${CXX_SOURCES})
endif()

if (DEFINED ENV{DUCC0_USE_NANOBIND})
    execute_process(
            COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE NB_DIR)
    execute_process(
            COMMAND "${Python_EXECUTABLE}" -m nanobind --include_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE NB_INCDIR)
    list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
    find_package(nanobind CONFIG REQUIRED)
    if (MULTIARCH)
        target_include_directories(${PKGNAME}_lib_v1 PRIVATE ${NB_INCDIR})
        target_include_directories(${PKGNAME}_lib_v3 PRIVATE ${NB_INCDIR})
        target_include_directories(${PKGNAME}_lib_v4 PRIVATE ${NB_INCDIR})
        nanobind_add_module(${PKGNAME} FREE_THREADED NOSTRIP NB_SUPPRESS_WARNINGS NOMINSIZE python/ducc_driver_multiarch.cc)
        target_link_libraries(${PKGNAME} PRIVATE ${PKGNAME}_lib_v1 ${PKGNAME}_lib_v3 ${PKGNAME}_lib_v4)
    else()
        nanobind_compile_options(${PKGNAME}_lib)
        nanobind_add_module(${PKGNAME} FREE_THREADED NOSTRIP NB_SUPPRESS_WARNINGS NOMINSIZE python/ducc_driver_single.cc)
        target_link_libraries(${PKGNAME} PRIVATE ${PKGNAME}_lib)
    endif()
    set (DEFS ${DEFS} DUCC0_USE_NANOBIND)
# Enable this if stubs are required
#    nanobind_add_stub(
#        ${PKGNAME}_stub
#        MODULE ${PKGNAME}
#        OUTPUT ${PKGNAME}.pyi
#        PYTHON_PATH $<TARGET_FILE_DIR:${PKGNAME}>
#        DEPENDS ${PKGNAME}
#        RECURSIVE
#    )
else()
    find_package(pybind11 REQUIRED)
    if (MULTIARCH)
        target_include_directories(${PKGNAME}_lib_v1 PRIVATE ${pybind11_INCLUDE_DIRS})
        target_include_directories(${PKGNAME}_lib_v3 PRIVATE ${pybind11_INCLUDE_DIRS})
        target_include_directories(${PKGNAME}_lib_v4 PRIVATE ${pybind11_INCLUDE_DIRS})
        pybind11_add_module(${PKGNAME} NO_EXTRAS python/ducc_driver_multiarch.cc)
        target_link_libraries(${PKGNAME} PRIVATE ${PKGNAME}_lib_v1 ${PKGNAME}_lib_v3 ${PKGNAME}_lib_v4)
    else()
        target_include_directories(${PKGNAME}_lib PRIVATE ${pybind11_INCLUDE_DIRS})
        pybind11_add_module(${PKGNAME} NO_EXTRAS python/ducc_driver_single.cc)
        target_link_libraries(${PKGNAME} PRIVATE ${PKGNAME}_lib)
    endif()
endif()

if (NOT MSVC)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "ppc|ppc64|powerpc|powerpc64" OR (APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES "ppc|ppc64"))
        # PowerPC arch does not have -march flag.
        set(DUCC0_ARCH_FLAGS "-mtune=native" CACHE STRING "Compiler flags for specifying target architecture.")
    elseif (CMAKE_SYSTEM_NAME MATCHES "Emscripten")
        set(DUCC0_ARCH_FLAGS "" CACHE STRING "Compiler flags for specifying target architecture.")
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
        -Wzero-as-null-pointer-constant -Wno-psabi)
#    set (DUCC0_OPT_FLAGS -O3 -ffp-contract=fast -fexcess-precision=fast -fno-math-errno -fno-signed-zeros -fno-trapping-math -fassociative-math -freciprocal-math)
    # we can use -ffast-math here because the flags will not be used for linking
    set (DUCC0_OPT_FLAGS -O3 -ffast-math)
    set (DUCC0_NOOPT_FLAGS -O0)
    set (DUCC0_DEBUG_FLAGS -g)
    set (DUCC0_STRIP_FLAGS -s)
endif()
SET (COMP_OPTS ${COMP_OPTS} ${DUCC0_WARN_FLAGS})

if (OPTFLAGS MATCHES "^native")
    SET (COMP_OPTS ${COMP_OPTS} ${DUCC0_ARCH_FLAGS})
endif()
if (OPTFLAGS MATCHES "-debug$")
    SET (COMP_OPTS ${COMP_OPTS} ${DUCC0_DEBUG_FLAGS})
elseif (OPTFLAGS MATCHES "-strip$")
    SET (LINK_OPTS ${LINK_OPTS} ${DUCC0_STRIP_FLAGS})
endif()
if (OPTFLAGS MATCHES "^none")
    SET (COMP_OPTS ${COMP_OPTS} ${DUCC0_NOOPT_FLAGS})
else()
    SET (COMP_OPTS ${COMP_OPTS} ${DUCC0_OPT_FLAGS})
endif()

if(DEFINED ENV{DUCC0_CFLAGS})
    separate_arguments(CFLAGS_TMP NATIVE_COMMAND "$ENV{DUCC0_CFLAGS}")
    SET (COMP_OPTS ${COMP_OPTS} ${CFLAGS_TMP})
endif()
if(DEFINED ENV{DUCC0_LFLAGS})
    separate_arguments(LFLAGS_TMP NATIVE_COMMAND "$ENV{DUCC0_LFLAGS}")
    SET (LINK_OPTS ${LINK_OPTS} ${LFLAGS_TMP})
endif()
if(DEFINED ENV{DUCC0_FLAGS})
    separate_arguments(FLAGS_TMP NATIVE_COMMAND "$ENV{DUCC0_FLAGS}")
    SET (COMP_OPTS ${COMP_OPTS} ${FLAGS_TMP})
    SET (LINK_OPTS ${LINK_OPTS} ${FLAGS_TMP})
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Emscripten")
    list(APPEND DEFS DUCC0_NO_LOWLEVEL_THREADING)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(${PKGNAME} PRIVATE Threads::Threads)
endif()

# settings for the final compile/link step
target_compile_definitions(${PKGNAME} PRIVATE ${DEFS} PKGNAME=${PKGNAME} PKGVERSION=${PKGVERSION})
set_property(TARGET ${PKGNAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
target_link_options(${PKGNAME} PRIVATE ${LINK_OPTS})
target_compile_features(${PKGNAME} PRIVATE cxx_std_17)

# setting s for the library part(s)
if (MULTIARCH)
    target_compile_definitions(${PKGNAME}_lib_v1 PRIVATE ${DEFS} DUCC0_NAMESPACE=${PKGNAME}_v1)
    target_compile_definitions(${PKGNAME}_lib_v3 PRIVATE ${DEFS} DUCC0_NAMESPACE=${PKGNAME}_v3)
    target_compile_definitions(${PKGNAME}_lib_v4 PRIVATE ${DEFS} DUCC0_NAMESPACE=${PKGNAME}_v4)
    target_compile_options(${PKGNAME}_lib_v1 PRIVATE ${COMP_OPTS} -march=x86-64)
    target_compile_options(${PKGNAME}_lib_v3 PRIVATE ${COMP_OPTS} -march=x86-64-v3)
    target_compile_options(${PKGNAME}_lib_v4 PRIVATE ${COMP_OPTS} -march=x86-64-v4)
    target_compile_features(${PKGNAME}_lib_v1 PRIVATE cxx_std_17)
    target_compile_features(${PKGNAME}_lib_v3 PRIVATE cxx_std_17)
    target_compile_features(${PKGNAME}_lib_v4 PRIVATE cxx_std_17)

    set_property(TARGET ${PKGNAME}_lib_v1 PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
    set_property(TARGET ${PKGNAME}_lib_v3 PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
    set_property(TARGET ${PKGNAME}_lib_v4 PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
else()
    target_compile_definitions(${PKGNAME}_lib PRIVATE ${DEFS} DUCC0_NAMESPACE=${PKGNAME})
    target_compile_features(${PKGNAME}_lib PRIVATE cxx_std_17)

    set_property(TARGET ${PKGNAME}_lib PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
endif()

install(TARGETS ${PKGNAME} LIBRARY DESTINATION .)
