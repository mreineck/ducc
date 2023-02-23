use ndarray::{Array1, ArrayView, ArrayViewMut, Dimension};
use num_complex::Complex;
use std::any::TypeId;
use std::ffi::c_void;
use std::mem::size_of;

// Questions:
//
// - How to unify mutslice2arrdesc and slice2arrdesc?

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

extern "C" {
    fn c2c_external(
        inp: &RustArrayDescriptor,
        out: &mut RustArrayDescriptor,
        axes: &RustArrayDescriptor,
        forward: bool,
        fct: f64,
        nthreads: usize,
    );
}

pub fn c2c<A: 'static, D: ndarray::Dimension>(
    inp: ArrayView<Complex<A>, D>,
    out: ArrayViewMut<Complex<A>, D>,
    axes: Vec<usize>,
    forward: bool,
    fct: f64,
    nthreads: usize,
) {
    let inp2 = slice2arrdesc(inp);
    let mut out2 = mutslice2arrdesc(out);
    let axes2 = Array1::from_vec(axes);
    let axes3 = slice2arrdesc(axes2.view());
    unsafe {
        c2c_external(&inp2, &mut out2, &axes3, forward, fct, nthreads);
    }
}

fn mutslice2arrdesc<'a, A: 'static, D: Dimension>(
    slc: ArrayViewMut<'a, A, D>,
) -> RustArrayDescriptor {
    let dtype: u8 = {
        if TypeId::of::<A>() == TypeId::of::<f64>() {
            7
        } else if TypeId::of::<A>() == TypeId::of::<f32>() {
            3
        } else if TypeId::of::<A>() == TypeId::of::<Complex<f64>>() {
            7 + 64
        } else if TypeId::of::<A>() == TypeId::of::<Complex<f32>>() {
            3 + 64
        } else {
            panic!("typeid not working");
        }
    };
    RustArrayDescriptor {
        ndim: slc.shape().len() as u8,
        dtype: dtype,
        shape: format_shape(slc.shape()),
        stride: format_stride(slc.strides()),
        data: slc.as_ptr() as *mut c_void,
    }
}

fn slice2arrdesc<'a, A: 'static, D: Dimension>(slc: ArrayView<'a, A, D>) -> RustArrayDescriptor {
    let dtype: u8 = {
        if TypeId::of::<A>() == TypeId::of::<f64>() || TypeId::of::<A>() == TypeId::of::<f32>() {
            (size_of::<A>() - 1) as u8
        } else if TypeId::of::<A>() == TypeId::of::<Complex<f64>>() {
            7 + 64
        } else if TypeId::of::<A>() == TypeId::of::<Complex<f32>>() {
            3 + 64
        } else if TypeId::of::<A>() == TypeId::of::<usize>() {
            (size_of::<A>() - 1 + 32) as u8
        } else {
            panic!("typeid not working");
        }
    };
    RustArrayDescriptor {
        ndim: slc.shape().len() as u8,
        dtype: dtype,
        shape: format_shape(slc.shape()),
        stride: format_stride(slc.strides()),
        data: slc.as_ptr() as *mut c_void,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use ndarray::{Array, Dim};
    use ndarray::prelude::*;

    #[test]
    fn square_test() {
        let shape = (2, 3, 3);
        let b = Array::from_elem(shape, Complex::<f64>::new(12., 0.));
        let mut c = Array::from_elem(shape, Complex::<f64>::new(0., 0.));
        println!("{:8.4}", b);
        let axes = vec![0, 2];
        c2c(b.view(), c.view_mut(), axes, true, 1., 1);
        println!("{:8.4}", c);
        panic!("asdf");
    }
}
