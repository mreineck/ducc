use ndarray::{Array1, ArrayView, ArrayViewMut, Dimension};
use num_complex::Complex;
use std::any::TypeId;
use std::ffi::c_void;
use std::mem::size_of;

use std::cell::UnsafeCell;

// Questions:
//
// - How to unify mutslice2arrdesc and slice2arrdesc?

// Debugging
// fn print_type_of<T>(_: &T) {
//     println!("{}", std::any::type_name::<T>())
// }
// /Debugging

// Related to RustArrayDescriptor
#[repr(C)]
pub struct RustArrayDescriptor {
    shape: [u64; 10], // TODO Make the "10" variable
    stride: [i64; 10],
    data: *mut c_void,
    ndim: u8,
    dtype: u8,
}

fn format_shape(ndinp: &[usize]) -> [u64; 10] {
    let mut res = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (i, elem) in ndinp.iter().enumerate() {
        res[i] = *elem as u64;
    }
    return res;
}

fn format_stride(ndinp: &[isize]) -> [i64; 10] {
    let mut res = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (i, elem) in ndinp.iter().enumerate() {
        res[i] = *elem as i64;
    }
    return res;
}

fn type2typeid<A: 'static>() -> u8 {
    if TypeId::of::<A>() == TypeId::of::<f64>() {
        7
    } else if TypeId::of::<A>() == TypeId::of::<f32>() {
        3
    } else if TypeId::of::<A>() == TypeId::of::<Complex<f64>>() {
        7 + 64
    } else if TypeId::of::<A>() == TypeId::of::<Complex<f32>>() {
        3 + 64
    } else if TypeId::of::<A>() == TypeId::of::<usize>() {
        (size_of::<A>() - 1 + 32) as u8
    } else {
        println!("{}", std::any::type_name::<A>());
        panic!("typeid not supported");
    }
}

fn mutslice2arrdesc<'a, A: 'static, D: Dimension>(
    slc: ArrayViewMut<'a, A, D>,
) -> RustArrayDescriptor {
    RustArrayDescriptor {
        ndim: slc.shape().len() as u8,
        dtype: type2typeid::<A>(),
        shape: format_shape(slc.shape()),
        stride: format_stride(slc.strides()),
        data: slc.as_ptr() as *mut c_void,
    }
}

fn slice2arrdesc<'a, A: 'static, D: Dimension>(slc: ArrayView<'a, A, D>) -> RustArrayDescriptor {
    RustArrayDescriptor {
        ndim: slc.shape().len() as u8,
        dtype: type2typeid::<A>(),
        shape: format_shape(slc.shape()),
        stride: format_stride(slc.strides()),
        data: slc.as_ptr() as *mut c_void,
    }
}
// /Related to RustArrayDescriptor

// Interface
extern "C" {
    fn fft_c2c_(
        inp: &RustArrayDescriptor,
        out: &mut RustArrayDescriptor,
        axes: &RustArrayDescriptor,
        forward: bool,
        fct: f64,
        nthreads: usize,
    );
}

pub fn fft_c2c<A: 'static, D: ndarray::Dimension>(
    inp: ArrayView<Complex<A>, D>,
    out: ArrayViewMut<Complex<A>, D>,
    axes: &Vec<usize>,
    forward: bool,
    fct: f64,
    nthreads: usize,
) {
    let inp2 = slice2arrdesc(inp);
    let mut out2 = mutslice2arrdesc(out);
    let axes2 = Array1::from_vec(axes.to_vec());
    let axes3 = slice2arrdesc(axes2.view());
    unsafe {
        fft_c2c_(&inp2, &mut out2, &axes3, forward, fct, nthreads);
    }
}

pub fn fft_c2c_inplace<A: 'static, D: ndarray::Dimension>(
    inpout: ArrayViewMut<Complex<A>, D>,
    axes: &Vec<usize>,
    forward: bool,
    fct: f64,
    nthreads: usize,
) {
    let inpout2 = UnsafeCell::new(mutslice2arrdesc(inpout));
    let axes2 = Array1::from_vec(axes.to_vec());
    let axes3 = slice2arrdesc(axes2.view());
    unsafe {
        fft_c2c_(& *inpout2.get(), &mut *inpout2.get(), &axes3, forward, fct, nthreads);
    }
}
// /Interface

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    // use ndarray::prelude::*;

    // TODO Write tests that go through all combinations of axes for 1d-3d, do FFT of arrays that
    // contain only ones, check if sums are consistent

    // TODO FFT back and forth with correct normalization and check that equal

    #[test]
    fn fft_test() {
        let shape = (2, 3, 3);

        let b = Array::from_elem(shape, Complex::<f64>::new(12., 0.));
        let mut c = Array::from_elem(shape, Complex::<f64>::new(0., 0.));
        println!("{:8.4}", b);
        let axes = vec![0, 2];
        fft_c2c(b.view(), c.view_mut(), &axes, true, 1., 1);
        println!("{:8.4}", c);

        fft_c2c_inplace(c.view_mut(), &axes, true, 1., 1);
    }
}
