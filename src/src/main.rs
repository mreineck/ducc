use ndarray::{s, Array, ArrayView, Dimension, Zip};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use num_complex::Complex;
use std::any::TypeId;
use std::ffi::c_void;

// TODO Error handling

#[repr(C)]
struct RustArrayDescriptor {
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
    fn square(inp: &mut RustArrayDescriptor);
    // fn c2c(inp: &RustArrayDescriptor, out: &mut RustArrayDescriptor);
    // fn c2c_inplace(inpout: &mut RustArrayDescriptor);
}

fn slice2ArrayDesc<'a, A: 'static, D: Dimension>(slc: ArrayView<'a, A, D>) -> RustArrayDescriptor {
    let dtype: u8 = {
        if TypeId::of::<A>() == TypeId::of::<f64>() {
            7
        } else if TypeId::of::<A>() == TypeId::of::<f32>() {
            3
        } else if TypeId::of::<A>() == TypeId::of::<Complex<f64>>() {
            7+64
        } else if TypeId::of::<A>() == TypeId::of::<Complex<f32>>() {
            3+64
        } else {
            panic!("typeid not working");
        }
    };
    // assert_eq!(D.ndim, slc.shape().len());

    RustArrayDescriptor {
        ndim: slc.shape().len() as u8,
        dtype: dtype,
        shape: format_shape(slc.shape()),
        stride: format_stride(slc.strides()),
        data: slc.as_ptr() as *mut c_void,
    }
}

fn main() {
    let a = Array::random((5, 6, 7), Uniform::new(0. as f32, 11. as f32));
    let slice = a.slice(s![.., 1, ..]);
    let mut arr = slice2ArrayDesc(slice);

    println!("{:8.4}", slice);
    unsafe {
        square(&mut arr);
    }
    println!("{:8.4}", slice);

    let m_re = Array::<f64, _>::random((5, 6), Uniform::new(-1.0, 1.0));
    let m_im = Array::<f64, _>::random((5, 6), Uniform::new(-1.0, 1.0));
    let mut m_c = Array::<Complex<f64>, _>::zeros((5, 6));
    Zip::from(&mut m_c)
        .and(&m_re)
        .and(&m_im)
        .for_each(|c, &re, &im| {
            *c = Complex { re, im };
        });

    let mut arr2 = slice2ArrayDesc(m_c.slice(s![.., ..]));  // TODO Simplify!
    println!("{:8.4}", m_c);
    unsafe {
        square(&mut arr2);
    }
    println!("{:8.4}", m_c);
}
