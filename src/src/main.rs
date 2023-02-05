use ndarray::{s, Array};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

// use num_complex::Complex;

#[cxx::bridge(namespace = "ducc0")]
mod ffi {
    struct RustArrayDescriptor {
        shape: [u64; 10],  // TODO Make the "10" variable
        stride: [i64; 10],
        ndim: u8,
        dtype: u8,
        data: *const f64
    }

    unsafe extern "C++" {
        include!("ducc0/bindings/array_descriptor.h");

        fn get_ndim(inp: &RustArrayDescriptor) -> u8;
        fn get_dtype(inp: &RustArrayDescriptor) -> u8;
        fn get_shape(desc: &RustArrayDescriptor, idim: u8) -> u64;
        fn get_stride(desc: &RustArrayDescriptor, idim: u8) -> i64;

        fn square(inp: &mut RustArrayDescriptor);
    }
}

fn format_shape(ndinp: &[usize]) -> [u64; 10] {
    let mut res = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (i, elem) in ndinp.iter().enumerate() {
        res[i] = *elem as u64;
    }
    return res
}

fn format_stride(ndinp: &[isize]) -> [i64; 10] {
    let mut res = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (i, elem) in ndinp.iter().enumerate() {
        res[i] = *elem as i64;
    }
    return res
}

fn main() {
    let a = Array::random((5, 6, 7), Uniform::new(0., 11.));
    let slice = a.slice(s![.., 1, ..]);

    let mut arr = ffi::RustArrayDescriptor {
        ndim: 2,
        dtype: 2,
        shape: format_shape(slice.shape()),
        stride: format_stride(slice.strides()),
        data: slice.as_ptr()
    };
    println!("{}", ffi::get_shape(&arr, 1));
    println!("{}", ffi::get_stride(&arr, 1));
    println!("{}", ffi::get_ndim(&arr));
    println!("{}", ffi::get_dtype(&arr));

    println!("{:?}", slice.shape());
    println!("{:?}", slice.strides());

    println!("{:8.4}", slice);
    ffi::square(&mut arr);
    println!("{:8.4}", slice);
}
