use ndarray::{ArrayView, Dimension, ArrayViewMut, array, ArrayBase, Array1};
use num_complex::Complex;
use std::any::TypeId;
use std::ffi::c_void;
use std::mem::size_of;

// TODO Error handling

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
    fn c_square(inp: &mut RustArrayDescriptor);
    fn c2c_double(
        inp: &RustArrayDescriptor,
        out: &mut RustArrayDescriptor,
        axes: &RustArrayDescriptor,
        forward: bool,
        fct: f64,
        nthreads: usize,
    );
    fn c2c_inplace_double(
        inout: &mut RustArrayDescriptor,
        axes: &RustArrayDescriptor,
        forward: bool,
        fct: f64,
        nthreads: usize,
    );
}

pub fn square<A: 'static, D: ndarray::Dimension>(inp: ArrayViewMut<A, D>) {
    let mut inp2 = mutslice2arrdesc(inp);
    unsafe {
        c_square(&mut inp2);
    }
}

// TODO proper handling of fct dtype
pub fn c2c<A: 'static, D: ndarray::Dimension>(inp: ArrayView<Complex<A>, D>, out: ArrayViewMut<Complex<A>, D>, axes: Vec<usize>,
    forward: bool, fct: f64, nthreads: usize ) {
    let inp2 = slice2arrdesc(inp);
    let mut out2 = mutslice2arrdesc(out);
    let axes2 = Array1::from_vec(axes);
    let axes3 = slice2arrdesc(axes2.view());
    // if TypeId::of::<A>() == TypeId::of::<f64>() {
    unsafe{
    c2c_double(&inp2, &mut out2, &axes3, forward, fct, nthreads);
    }
    // }
    // else {
            // panic!("typeid not working");
    // }

}

fn mutslice2arrdesc<'a, A: 'static, D: Dimension>(slc: ArrayViewMut<'a, A, D>) -> RustArrayDescriptor {
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
        if TypeId::of::<A>() == TypeId::of::<f64>() || TypeId::of::<A>() == TypeId::of::<f32>()
        {
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

// fn main() {
//     let a = Array::random((5, 6, 7), Uniform::new(0. as f32, 11. as f32));
//     let slice = a.slice(s![.., 1, ..]);
//     let mut arr = slice2arrdesc(slice);
//
//     println!("{:8.4}", slice);
//     unsafe {
//         square(&mut arr);
//     }
//     println!("{:8.4}", slice);
//
//     let m_re = Array::<f64, _>::random((5, 6), Uniform::new(-1.0, 1.0));
//     let m_im = Array::<f64, _>::random((5, 6), Uniform::new(-1.0, 1.0));
//     let mut m_c = Array::<Complex<f64>, _>::zeros((5, 6));
//     Zip::from(&mut m_c)
//         .and(&m_re)
//         .and(&m_im)
//         .for_each(|c, &re, &im| {
//             *c = Complex { re, im };
//         });
//
//     let mut arr2 = slice2arrdesc(m_c.slice(s![.., ..])); // TODO Simplify!
//     println!("{:8.4}", m_c);
//     unsafe {
//         square(&mut arr2);
//     }
//     println!("{:8.4}", m_c);
// }

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{s, Array, IxDyn, Zip};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn square_test() {
        let shape = (2, 3, 3);
        let mut a = Array::random(shape, Uniform::new(-1., 10.));

        // println!("{:8.4}", a);
        // square(a.view_mut());
        // println!("{:8.4}", a);

        let b = Array::ones(shape);
        let mut c = Array::ones(shape);
        let axes = vec![0, 2];
        println!("{:8.4}", b);
        c2c(b.view(), c.view_mut(), axes, true, 1., 1);
        println!("{:8.4}", c);

        panic!("asdf");

    }
}
