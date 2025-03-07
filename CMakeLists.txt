cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# Set the package name
set(PKGNAME ${SKBUILD_PROJECT_NAME})
# Set the version
set(PKGVERSION ${SKBUILD_PROJECT_VERSION})

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")

find_package(nanobind CONFIG REQUIRED)

set(CXX_SOURCES
        src/ducc0/healpix/healpix_base.cc
        src/ducc0/healpix/healpix_tables.cc
        src/ducc0/math/gl_integrator.cc
        src/ducc0/math/pointing.cc
        src/ducc0/math/gridding_kernel.cc
        src/ducc0/math/geom_utils.cc
        src/ducc0/math/wigner3j.cc
        src/ducc0/math/space_filling.cc
        src/ducc0/wgridder/wgridder.cc
        src/ducc0/infra/string_utils.cc
        src/ducc0/infra/threading.cc
        src/ducc0/infra/mav.cc
        src/ducc0/sht/sht.cc
)
set(PYMOD_SOURCES
        python/ducc.cc
        python/fft_pymod.cc
        python/healpix_pymod.cc
        python/misc_pymod.cc
        python/nufft_pymod.cc
        python/pointingprovider_pymod.cc
        python/sht_pymod.cc
        python/totalconvolve_pymod.cc
        python/wgridder_pymod.cc
)

nanobind_add_module(ducc0 NOMINSIZE ${CXX_SOURCES} ${PYMOD_SOURCES})

target_compile_definitions(ducc0 PRIVATE DUCC0_USE_NANOBIND)
target_compile_definitions(ducc0 PRIVATE PKGNAME=${PKGNAME} PKGVERSION=${PKGVERSION})

if (NOT MSVC)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "ppc|ppc64|powerpc|powerpc64" OR (APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES "ppc|ppc64"))
        # PowerPC arch does not have -march flag.
        set(DUCC_ARCH_FLAGS "-mtune=native" CACHE STRING "Compiler flags for specifying target architecture.")
    else ()
        set(DUCC_ARCH_FLAGS "-march=native" CACHE STRING "Compiler flags for specifying target architecture.")
    endif ()
    message(STATUS "Using GCC/Clang flags: ${DUCC_ARCH_FLAGS}")
else ()
    # Check for AVX, AVX512 and SSE support
    message(STATUS "Checking for AVX, AVX512 and SSE support")
    try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
            ${CMAKE_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckAVX.cpp
            COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT
            RUN_OUTPUT_VARIABLE RUN_OUTPUT)
    if (RUN_OUTPUT MATCHES "AVX512")
        set(DUCC_ARCH_FLAGS "/arch:AVX512" CACHE STRING "Compiler flags for specifying target architecture.")
    elseif (RUN_OUTPUT MATCHES "AVX")
        set(DUCC_ARCH_FLAGS "/arch:AVX" CACHE STRING "Compiler flags for specifying target architecture.")
    elseif (RUN_OUTPUT MATCHES "SSE")
        set(DUCC_ARCH_FLAGS "/arch:SSE" CACHE STRING "Compiler flags for specifying target architecture.")
    else ()
        set(DUCC_ARCH_FLAGS "" CACHE STRING "Compiler flags for specifying target architecture.")
    endif ()
    message(STATUS "CPU supports: ${RUN_OUTPUT}")
    message(STATUS "Using MSVC flags: ${DUCC_ARCH_FLAGS}")
endif ()

option(DUCC_USE_THREADS "Use threads for parallelization" ON)

include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/src)

target_compile_options(ducc0 PRIVATE ${DUCC_ARCH_FLAGS})

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(ducc0 PRIVATE
            -Wfatal-errors
            -Wfloat-conversion
            -W
            -Wall
            -Wstrict-aliasing
            -Wwrite-strings
            -Wredundant-decls
            -Woverloaded-virtual
            -Wcast-qual
            -Wcast-align
            -Wpointer-arith
            -Wnon-virtual-dtor
            -Wzero-as-null-pointer-constant
    )
endif()

if (DUCC_USE_THREADS)
    find_package(Threads REQUIRED)
    target_link_libraries(ducc0 PRIVATE Threads::Threads)
endif ()

target_compile_features(ducc0 PRIVATE cxx_std_17)

install(TARGETS ducc0 LIBRARY DESTINATION .)
