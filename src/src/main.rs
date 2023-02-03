use ndarray::{s, Array};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

// use num_complex::Complex;

#[cxx::bridge(namespace = "ducc0")]
mod ffi {
    struct RustArrayDescriptor {
        shape: [u64; 5],
        //stride: [i64; 5],
        ndim: u8,
        //dtype: u8,
        //data: *mut c_void
    }

    unsafe extern "C++" {
        include!("ducc0/bindings/array_descriptor.h");

        // type ArrayDescriptor;
        // fn getSimpleDescriptor(ndim: u8) -> UniquePtr<SimpleDescriptor>;
        // type SimpleDescriptor;

        fn get_ndim(inp: &RustArrayDescriptor) -> u8;
        fn set_ndim(desc: &mut RustArrayDescriptor, n: u8);

        fn get_shape(desc: &RustArrayDescriptor, idim: u8) -> u64;
        // fn set_shape(desc: &mut RustArrayDescriptor, idim: u8, val: u64);
    }
}

fn main() {
    // let a = Array::random((5, 6, 7), Uniform::new(0., 10.));
    // let slice = a.slice(s![.., 1, ..]);
    // println!("Whole array");
    // println!("{:8.4}", a);
    // println!("Slice");
    // println!("{:8.4}", slice);

    let mut arr = ffi::RustArrayDescriptor {
        ndim: 2,
        shape: [5, 6, 7, 0, 0],
    };
    println!("{}", ffi::get_ndim(&arr));

    println!("Before {}", ffi::get_ndim(&arr));
    ffi::set_ndim(&mut arr, 3);
    println!("After {}", ffi::get_ndim(&arr));

    println!("Shape Before {}", ffi::get_shape(&arr, 1));
}
