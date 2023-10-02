module ducc0
  use iso_c_binding
  implicit none
  private

  public :: fft_c2c_inplace

  interface
    subroutine fft_c2c_ll(in, out, axes, forward, fct, nthreads) bind(c, name="fft_c2c_c")
      use iso_c_binding
      type(*), intent(in) :: in(..)
      type(*), intent(inout) :: out(..)
      type(*), intent(in) :: axes(:)
      integer(C_INT), intent(in), value :: forward
      real(C_DOUBLE), intent(in), value :: fct
      integer(C_SIZE_T), intent(in), value :: nthreads
    end subroutine
  end interface

contains
  function cbool(arg) result(res)
    logical, intent(in) :: arg
    integer(C_INT) :: res
    res = merge(1_C_INT, 0_C_INT, arg)
  end function
  function csizet(arg) result(res)
    integer, intent(in) :: arg
    integer(C_SIZE_T) :: res
    res = int(arg, C_SIZE_T)
  end function
  function caxes(arg) result(res)
    integer, intent(in) :: arg(:)
    integer(C_SIZE_T) :: res(size(arg,1))
    res = int(arg, C_SIZE_T)
  end function

  subroutine fft_c2c_inplace(inout, axes, fwd, fct, nthreads)
    type(*), intent(inout) :: inout(:)
    integer, intent(in) :: axes(:)
    logical, intent(in) :: fwd
    real(C_DOUBLE), intent(in) :: fct
    integer,intent(in) :: nthreads

    call fft_c2c_ll(inout, inout, caxes(axes), cbool(fwd), fct, csizet(nthreads))
  end subroutine
end module

program blah
use ducc0
implicit none
complex(8) :: arr, arr2(10)
integer i
arr2(:)=0
arr2(1) = 1
call fft_c2c_inplace(arr2, (/1/), .true., 1.D0, 1)
do i = 1,10
  print *,arr2(i)
end do
call fft_c2c_inplace(arr2, (/1/), .false., 1.D0, 1)
do i = 1,10
  print *,arr2(i)
end do
!call fft_c2c_inplace(arr, (/1/),.false., 1.D0, 1)
end program
