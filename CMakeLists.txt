# TODO:
# - make multiarch work on Windows x86_64? Do we care enough?
# - make multiarch work on ARM?
cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION True)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_STANDARD 17)

if(DEFINED ENV{DUCC0_OPTIMIZATION})
    set (OPTFLAGS "$ENV{DUCC0_OPTIMIZATION}")
else()
    set (OPTFLAGS "native-strip")
endif()
if (OPTFLAGS MATCHES "^multiarch"
    AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
    AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    set (MULTIARCH ON)
else()
    set (MULTIARCH OFF)
endif()

if (MULTIARCH)
    function(ducc0_check_multiarch_global_flags source flags)
        if ("${flags}" STREQUAL "")
            return()
        endif()
        separate_arguments(_tokens NATIVE_COMMAND "${flags}")
        foreach(_flag IN LISTS _tokens)
            string(TOLOWER "${_flag}" _lower_flag)
            if (_lower_flag MATCHES "^-march=")
                if (NOT _lower_flag STREQUAL "-march=x86-64")
                    message(FATAL_ERROR
                        "Multiarch profile contamination: ${source} contains ${_flag}; only -march=x86-64 is allowed globally.")
                endif()
            elseif (_lower_flag MATCHES "^-mcpu="
                    OR _lower_flag MATCHES "^-m(no-)?(avx|sse|fma|bmi|lzcnt|f16c|movbe)"
                    OR _lower_flag MATCHES "^-m(no-)?(cx16|sahf|popcnt|aes|pclmul|sha|gfni|vaes|vpclmul|amx|abm|adx|rtm|hle|rdrnd|rdseed|clflushopt|clwb|waitpkg|serialize)"
                    OR _lower_flag MATCHES "^/arch:")
                message(FATAL_ERROR
                    "Multiarch profile contamination: ${source} contains ${_flag}; ISA flags must be set per profile.")
            elseif (_lower_flag MATCHES "^-flto($|=)"
                    OR _lower_flag STREQUAL "/gl")
                message(FATAL_ERROR
                    "Multiarch profile contamination: ${source} contains ${_flag}; global LTO is not allowed.")
            endif()
        endforeach()
    endfunction()

    set(_multiarch_flag_variables
        CMAKE_C_FLAGS
        CMAKE_CXX_FLAGS
        CMAKE_EXE_LINKER_FLAGS
        CMAKE_SHARED_LINKER_FLAGS
        CMAKE_MODULE_LINKER_FLAGS)
    foreach(_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        list(APPEND _multiarch_flag_variables
            CMAKE_C_FLAGS_${_config}
            CMAKE_CXX_FLAGS_${_config}
            CMAKE_EXE_LINKER_FLAGS_${_config}
            CMAKE_SHARED_LINKER_FLAGS_${_config}
            CMAKE_MODULE_LINKER_FLAGS_${_config})
    endforeach()
    foreach(_variable IN LISTS _multiarch_flag_variables)
        if (DEFINED ${_variable})
            ducc0_check_multiarch_global_flags(
                "${_variable}" "${${_variable}}")
        endif()
    endforeach()

    foreach(_variable CFLAGS CPPFLAGS CXXFLAGS LDFLAGS
                      DUCC0_CFLAGS DUCC0_FLAGS DUCC0_LFLAGS)
        set(_value "$ENV{${_variable}}")
        if (NOT "${_value}" STREQUAL "")
            ducc0_check_multiarch_global_flags(
                "environment variable ${_variable}" "${_value}")
        endif()
    endforeach()
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
    set (DEFS ${DEFS} DUCC0_MULTIARCH)
    add_library(${PKGNAME}_lib_v1 OBJECT ${PYMOD_SOURCES} ${CXX_SOURCES})
    add_library(${PKGNAME}_lib_v3 OBJECT ${PYMOD_SOURCES} ${CXX_SOURCES})
    add_library(${PKGNAME}_lib_v4 OBJECT ${PYMOD_SOURCES} ${CXX_SOURCES})
    set_property(TARGET ${PKGNAME}_lib_v1 PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    set_property(TARGET ${PKGNAME}_lib_v3 PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    set_property(TARGET ${PKGNAME}_lib_v4 PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    foreach(_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_property(TARGET ${PKGNAME}_lib_v1
            PROPERTY INTERPROCEDURAL_OPTIMIZATION_${_config} FALSE)
        set_property(TARGET ${PKGNAME}_lib_v3
            PROPERTY INTERPROCEDURAL_OPTIMIZATION_${_config} FALSE)
        set_property(TARGET ${PKGNAME}_lib_v4
            PROPERTY INTERPROCEDURAL_OPTIMIZATION_${_config} FALSE)
    endforeach()
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
    set(INC_DIRS ${INC_DIRS} ${NB_INCDIR} ${Python_INCLUDE_DIRS})
    if (MULTIARCH)
        nanobind_compile_options(${PKGNAME}_lib_v1)
        nanobind_compile_options(${PKGNAME}_lib_v3)
        nanobind_compile_options(${PKGNAME}_lib_v4)
    else()
        nanobind_compile_options(${PKGNAME}_lib)
    endif()
    nanobind_add_module(${PKGNAME}
        FREE_THREADED NOSTRIP NB_SUPPRESS_WARNINGS NOMINSIZE
        python/ducc_driver.cc python/multiarch.cc)
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
    set(INC_DIRS ${INC_DIRS} ${pybind11_INCLUDE_DIRS})
    pybind11_add_module(${PKGNAME} NO_EXTRAS
        python/ducc_driver.cc python/multiarch.cc)
endif()

if (MULTIARCH)
    set(DUCC0_ARCH_FLAGS "")
    message(STATUS "Using per-profile x86-64 flags for multiarch build")
elseif (NOT MSVC)
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
    SET (DEFS ${DEFS} DUCC0_NO_LOWLEVEL_THREADING)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(${PKGNAME} PRIVATE Threads::Threads)
endif()

SET (DEFS ${DEFS} PKGVERSION=${PKGVERSION})

# settings for the final compile/link step
target_compile_definitions(${PKGNAME} PRIVATE ${DEFS} PKGNAME=${PKGNAME})
if (MULTIARCH)
    target_compile_options(${PKGNAME} PRIVATE ${COMP_OPTS} -march=x86-64)
else()
    target_compile_options(${PKGNAME} PRIVATE ${COMP_OPTS})
endif()
target_link_options(${PKGNAME} PRIVATE ${LINK_OPTS})

# setting s for the library part(s)
if (MULTIARCH)
    target_include_directories(${PKGNAME}_lib_v1 SYSTEM PRIVATE ${INC_DIRS})
    target_include_directories(${PKGNAME}_lib_v3 SYSTEM PRIVATE ${INC_DIRS})
    target_include_directories(${PKGNAME}_lib_v4 SYSTEM PRIVATE ${INC_DIRS})
    target_compile_definitions(${PKGNAME}_lib_v1 PRIVATE ${DEFS} DUCC0_NAMESPACE=${PKGNAME}_v1)
    target_compile_definitions(${PKGNAME}_lib_v3 PRIVATE ${DEFS} DUCC0_NAMESPACE=${PKGNAME}_v3)
    target_compile_definitions(${PKGNAME}_lib_v4 PRIVATE ${DEFS} DUCC0_NAMESPACE=${PKGNAME}_v4)
    target_compile_options(${PKGNAME}_lib_v1 PRIVATE ${COMP_OPTS} -march=x86-64)
    target_compile_options(${PKGNAME}_lib_v3 PRIVATE ${COMP_OPTS} -march=x86-64-v3)
    target_compile_options(${PKGNAME}_lib_v4 PRIVATE ${COMP_OPTS} -march=x86-64-v4)
    target_link_libraries(${PKGNAME} PRIVATE ${PKGNAME}_lib_v1 ${PKGNAME}_lib_v3 ${PKGNAME}_lib_v4)
else()
    target_include_directories(${PKGNAME}_lib SYSTEM PRIVATE ${INC_DIRS})
    target_compile_definitions(${PKGNAME}_lib PRIVATE ${DEFS} DUCC0_NAMESPACE=${PKGNAME})
    target_compile_options(${PKGNAME}_lib PRIVATE ${COMP_OPTS})
    target_link_libraries(${PKGNAME} PRIVATE ${PKGNAME}_lib)
endif()

install(TARGETS ${PKGNAME} LIBRARY DESTINATION .)
