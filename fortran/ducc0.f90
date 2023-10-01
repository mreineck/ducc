module ducc0
  use iso_c_binding
  implicit none
  private

  public :: fft_c2c_inplace, arr2desc, descriptor

  INTEGER, PARAMETER :: COMPLEX64  = KIND((1.0_C_FLOAT, 1.0_C_FLOAT))
  INTEGER, PARAMETER :: COMPLEX128 = KIND((1.0_C_DOUBLE, 1.0_C_DOUBLE))

  type, bind(c) :: descriptor
    integer(C_INT64_T) :: shape(10)
    integer(C_INT64_T) :: stride(10)
    type(C_PTR) :: data
    integer(C_INT8_T) :: ndim
    integer(C_INT8_T) :: dtype
  end type descriptor

  interface
    subroutine print_array(desc) bind(c, name="print_array")
      import:: descriptor
      type(descriptor), intent(in) :: desc
    end subroutine print_array
    function get_stride(p1, p2, dtype) bind(c, name="get_stride") result(res)
      use iso_c_binding
      integer(C_PTRDIFF_T) :: res
      type(C_PTR), intent(in), value :: p1, p2
      integer(C_INT8_T), intent(in), value :: dtype
    end function get_stride
    subroutine fft_c2c_inplace_ll(inout, axes, forward, fct, nthreads) bind(c, name="fft_c2c_inplace")
      use iso_c_binding
      import:: descriptor
      type(descriptor), intent(in) :: inout
      type(descriptor), intent(in) :: axes
      integer(C_INT), intent(in), value :: forward
      real(C_DOUBLE), intent(in), value :: fct
      integer(C_SIZE_T), intent(in), value :: nthreads
    end subroutine
    subroutine arr2desc_c(arr, desc) bind(c, name="arraytodesc_c")
      import:: descriptor
      type(*), intent(in) :: arr(..)
      type(descriptor), intent(out) :: desc
    end subroutine
  end interface

  interface fft_c2c_inplace
    module procedure fft_c2c_c128_1d_inplace, fft_c2c_c128_2d_inplace
  end interface

contains
  function cbool(arg) result(res)
    logical, intent(in) :: arg
    integer(C_INT) :: res
    res = merge(1_C_INT, 0_C_INT, arg)
  end function

  function sizetarr(arg) result(res)
    integer, intent(in) :: arg(:)
    integer(C_SIZE_T) :: res(size(arg,1))
    integer i
    do i = 1, size(arg,1)
      res(i) = int(arg(i), C_SIZE_T)
      print *, i, arg(i), res(i)
    end do
  end function

  subroutine fft_c2c_c128_1d_inplace(inout, fwd, fct, nthreads)
    complex(COMPLEX128), intent(inout) :: inout(:)
    logical, intent(in) :: fwd
    real(C_DOUBLE), intent(in) :: fct
    integer,intent(in) :: nthreads

    integer(C_SIZE_T), parameter :: axes(1) = (/1/)
    call fft_c2c_inplace_ll(arr2desc(inout), arr2desc_u(axes), cbool(fwd), fct, int(nthreads, C_SIZE_T))
  end subroutine
  subroutine fft_c2c_c128_2d_inplace(inout, axes, fwd, fct, nthreads)
    complex(COMPLEX128), intent(inout) :: inout(:,:)
    integer, intent(in) :: axes(:)
    logical, intent(in) :: fwd
    real(C_DOUBLE), intent(in) :: fct
    integer,intent(in) :: nthreads

    integer(C_SIZE_T), dimension(:) :: axes2(size(axes,1))
    axes2(:) = axes(:)
    call fft_c2c_inplace_ll(arr2desc(inout), arr2desc_u(axes2), cbool(fwd), fct, int(nthreads, C_SIZE_T))
  end subroutine
  function arr2desc(arr) result(res)
    type(*), intent(in) :: arr(..)
    type(descriptor) :: res
    call arr2desc_c(arr, res)
  end function
  function arr2desc_u(arr) result(res)
    type(*), intent(in) :: arr(..)
    type(descriptor) :: res
    call arr2desc_c(arr, res)
    res%dtype = res%dtype+16
  end function
end module

program blah
use ducc0
implicit none
double precision :: arr, arr2(10)
type(descriptor) :: d1
integer i
d1 = arr2desc(arr)
arr2(:)=0
arr2(1) = 1
print *, d1%dtype
print *, d1%ndim
print *, d1%shape(1)
print *, d1%shape(2)
print *, d1%stride(1)
print *, d1%stride(2)
!call fft_c2c_inplace(arr2, .true., 1.D0, 1)
!do i = 1,10
!  print *,arr2(i)
!end do
!call fft_c2c_inplace(arr2, .false., 1.D0, 1)
!do i = 1,10
!  print *,arr2(i)
!end do
!call fft_c2c_inplace(arr, (/1/),.false., 1.D0, 1)
end program
