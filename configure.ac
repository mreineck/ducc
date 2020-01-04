AC_INIT([libmrutil], [1.0.0])
AM_INIT_AUTOMAKE([foreign subdir-objects -Wall -Werror])
AM_MAINTAINER_MODE([enable])


dnl
dnl Needed for linking on Windows.
dnl Protect with m4_ifdef because AM_PROG_AR is required in
dnl autoconf >= 1.12 when using -Wall, but the macro is
dnl absent in old versions of autoconf.
dnl
m4_ifdef([AM_PROG_AR], [AM_PROG_AR])

LT_INIT
AC_CONFIG_MACRO_DIR([m4])

dnl
dnl Enable silent build rules if this version of Automake supports them
dnl
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])


AC_PROG_CXX
AX_CXX_COMPILE_STDCXX([17])

AC_PROG_LIBTOOL

tmpval=`echo $CXXFLAGS | grep -c '\-DMULTIARCH'`
AM_CONDITIONAL([HAVE_MULTIARCH], [test $tmpval -gt 0])

PACKAGE_LIBS="-lmrutil"

dnl
dnl Create pkgconfig .pc file.
dnl
AX_CREATE_PKGCONFIG_INFO(,,,,[])
AC_SUBST([AM_CXXFLAGS])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
