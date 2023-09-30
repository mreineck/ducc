module ducc0
  use iso_c_binding
  implicit none
  private

  public :: fft_c2c_inplace

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
  end interface

  interface arr2desc
    module procedure a2d_f32_1, a2d_f32_2, &
                     a2d_f64_1, a2d_f64_2, &
                     a2d_c64_1, a2d_c64_2, &
                     a2d_c128_1, a2d_c128_2, &
                     a2d_i64_1
  end interface
  interface arr2desc_u
    module procedure a2d_i64_1_u
  end interface

  interface get_dtype
    module procedure get_dtype_f32, get_dtype_f64, &
                     get_dtype_c64, get_dtype_c128, &
                     get_dtype_i64
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

  subroutine do_common_stuff(dtype, ptr, shp, desc)
    integer(C_INT8_T), intent(in) :: dtype
    type(C_PTR), intent(in) :: ptr
    integer, intent(in) :: shp(:)
    type(descriptor), intent(inout) :: desc
    integer :: i

    ! assert that ndim<=10 !
    desc%dtype = dtype
    desc%data = ptr
    desc%ndim = int(size(shp,1), kind=C_INT8_T)
    do i = 1, desc%ndim
      desc%shape(i) = shp(i)
    enddo
  end subroutine

  function a2d_f32_1 (arr) result(res)
    real(C_FLOAT), intent(in), target, dimension(:) :: arr
    type(descriptor) :: res
    call do_common_stuff(get_dtype(arr), c_loc(arr), shape(arr), res)
    res%stride(1) = get_stride(res%data, c_loc(arr(min(size(arr,1),2))), res%dtype)
  end function
  function a2d_f32_2 (arr) result(res)
    real(C_FLOAT), intent(in), target, dimension(:,:) :: arr
    type(descriptor) :: res
    call do_common_stuff(get_dtype(arr), c_loc(arr), shape(arr), res)
    res%stride(1) = get_stride(res%data, c_loc(arr(min(size(arr,1),2),1)), res%dtype)
    res%stride(2) = get_stride(res%data, c_loc(arr(1,min(size(arr,2),2))), res%dtype)
  end function
  function a2d_f64_1 (arr) result(res)
    real(C_DOUBLE), intent(in), target, dimension(:) :: arr
    type(descriptor) :: res
    call do_common_stuff(get_dtype(arr), c_loc(arr), shape(arr), res)
    res%stride(1) = get_stride(res%data, c_loc(arr(min(size(arr,1),2))), res%dtype)
  end function
  function a2d_f64_2 (arr) result(res)
    real(C_DOUBLE), intent(in), target, dimension(:,:) :: arr
    type(descriptor) :: res
    call do_common_stuff(get_dtype(arr), c_loc(arr), shape(arr), res)
    res%stride(1) = get_stride(res%data, c_loc(arr(min(size(arr,1),2),1)), res%dtype)
    res%stride(2) = get_stride(res%data, c_loc(arr(1,min(size(arr,2),2))), res%dtype)
  end function
  function a2d_c64_1 (arr) result(res)
    complex(COMPLEX64), intent(in), target, dimension(:) :: arr
    type(descriptor) :: res
    call do_common_stuff(get_dtype(arr), c_loc(arr), shape(arr), res)
    res%stride(1) = get_stride(res%data, c_loc(arr(min(size(arr,1),2))), res%dtype)
  end function
  function a2d_c64_2 (arr) result(res)
    complex(COMPLEX64), intent(in), target, dimension(:,:) :: arr
    type(descriptor) :: res
    call do_common_stuff(get_dtype(arr), c_loc(arr), shape(arr), res)
    res%stride(1) = get_stride(res%data, c_loc(arr(min(size(arr,1),2),1)), res%dtype)
    res%stride(2) = get_stride(res%data, c_loc(arr(1,min(size(arr,2),2))), res%dtype)
  end function
  function a2d_c128_1 (arr) result(res)
    complex(COMPLEX128), intent(in), target, dimension(:) :: arr
    type(descriptor) :: res
    call do_common_stuff(get_dtype(arr), c_loc(arr), shape(arr), res)
    res%stride(1) = get_stride(res%data, c_loc(arr(min(size(arr,1),2))), res%dtype)
  end function
  function a2d_c128_2 (arr) result(res)
    complex(COMPLEX128), intent(in), target, dimension(:,:) :: arr
    type(descriptor) :: res
    call do_common_stuff(get_dtype(arr), c_loc(arr), shape(arr), res)
    res%stride(1) = get_stride(res%data, c_loc(arr(min(size(arr,1),2),1)), res%dtype)
    res%stride(2) = get_stride(res%data, c_loc(arr(1,min(size(arr,2),2))), res%dtype)
  end function
  function a2d_i64_1 (arr) result(res)
    integer(C_INT64_T), intent(in), target, dimension(:) :: arr
    type(descriptor) :: res
    call do_common_stuff(get_dtype(arr), c_loc(arr), shape(arr), res)
    res%stride(1) = get_stride(res%data, c_loc(arr(min(size(arr,1),2))), res%dtype)
  end function
  function a2d_i64_1_u (arr) result(res)
    integer(C_INT64_T), intent(in), target, dimension(:) :: arr
    type(descriptor) :: res
    call do_common_stuff(get_dtype(arr)+16_C_INT8_T, c_loc(arr), shape(arr), res)
    res%stride(1) = get_stride(res%data, c_loc(arr(min(size(arr,1),2))), res%dtype)
  end function

  function get_dtype_f32 (arr) result(res)
    real(C_FLOAT), intent(in) :: arr(..)
    integer(C_INT8_T) :: res
    if (.false.) print *, shape(arr)
    res = 3
  end function
  function get_dtype_f64 (arr) result(res)
    real(C_DOUBLE), intent(in) :: arr(..)
    integer(C_INT8_T) :: res
    if (.false.) print *, shape(arr)
    res = 7
  end function
  function get_dtype_c64 (arr) result(res)
    complex(COMPLEX64), intent(in) :: arr(..)
    integer(C_INT8_T) :: res
    if (.false.) print *, shape(arr)
    res = 67
  end function
  function get_dtype_c128 (arr) result(res)
    complex(COMPLEX128), intent(in) :: arr(..)
    integer(C_INT8_T) :: res
    if (.false.) print *, shape(arr)
    res = 71
  end function
  function get_dtype_i64 (arr) result(res)
    integer(C_INT64_T), intent(in) :: arr(..)
    integer(C_INT8_T) :: res
    if (.false.) print *, shape(arr)
    res = 23
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
end module

program blah
use ducc0
complex(8) :: arr(1:1,4:9), arr2(10)
integer i
arr2(:)=0
arr2(1) = 1
call fft_c2c_inplace(arr2, .true., 1.D0, 1)
do i = 1,10
  print *,arr2(i)
end do
call fft_c2c_inplace(arr2, .false., 1.D0, 1)
do i = 1,10
  print *,arr2(i)
end do
call fft_c2c_inplace(arr, (/1/),.false., 1.D0, 1)
end program
