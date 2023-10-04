! This code is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 2 of the License, or
! (at your option) any later version.
!
! This code is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License
! along with this code; if not, write to the Free Software
! Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

! Copyright (C) 2023 Max-Planck-Society
! Author: Martin Reinecke

module ducc0
  use iso_c_binding
  implicit none
  private

  public :: fft_c2c_inplace

  interface
    subroutine fft_c2c_c(in, out, axes, forward, fct, nthreads) bind(c, name="fft_c2c_c")
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
    type(*), intent(inout) :: inout(..)
    integer, intent(in) :: axes(:)
    logical, intent(in) :: fwd
    real(C_DOUBLE), intent(in) :: fct
    integer,intent(in) :: nthreads

    call fft_c2c_c(inout, inout, caxes(axes), cbool(fwd), fct, csizet(nthreads))
  end subroutine
end module
