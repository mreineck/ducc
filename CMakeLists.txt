cmake_minimum_required(VERSION 3.10)
# set the project name
project(DUCC)


# Set the package name
set(PKGNAME "ducc0")
# Set the version
find_package(Git)
if (GIT_FOUND)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} describe --abbrev=0 --tags
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else ()
    message(WARNING "Git not found! Using default version unknown")
    set(VERSION "unknown")
endif ()

if (NOT MSVC)
    set(DUCC_ARCH_FLAGS "-march=native" CACHE STRING "Compiler flags for specifying target architecture.")
else ()
    set(DUCC_ARCH_FLAGS "" CACHE STRING "Compiler flags for specifying target architecture.")
endif ()

option(DUCC_USE_THREADS "Use threads for parallelization" ON)

add_library(ducc0 STATIC
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
        src/ducc0/infra/communication.cc
        src/ducc0/infra/types.cc
        src/ducc0/infra/system.cc
        src/ducc0/infra/threading.cc
        src/ducc0/infra/mav.cc
        # add more source files here
)

target_compile_features(ducc0 PRIVATE cxx_std_17)
target_include_directories(ducc0 PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_include_directories(ducc0 SYSTEM INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/src>)
target_compile_options(ducc0 PRIVATE -ffast-math)
target_compile_options(ducc0 PRIVATE SHELL:$<$<CONFIG:Release,RelWithDebInfo>:${DUCC_ARCH_FLAGS}>)


target_compile_definitions(ducc0 PRIVATE PKGNAME=${PKGNAME} PKGVERSION=${VERSION})
if(MSVC)
    target_compile_options(ducc0 PRIVATE
            /W4 # Equivalent of -Wall
            /WX # Equivalent of -Wfatal-errors
            # Add other MSVC specific flags here
    )
endif()
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

install(TARGETS ducc0
        EXPORT ${PKGNAME}Targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )
# Find Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
    # Note the option ALL which allows to build the docs together with the application
    add_custom_target(doc_doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} src/doc/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM)
else(DOXYGEN_FOUND)
    message("Doxygen need to be installed to generate the doxygen documentation")
endif(DOXYGEN_FOUND)