use ndarray::{Array, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

// use num_complex::Complex;


#[cxx::bridge(namespace = "ducc0")]
mod ffi {
    // struct RustArrayDescriptor {
    //     shape: [u64; 10],
    //     stride: [i64; 10],
    //     ndim: u8,
    //     dtype: u8,
    //     //data: *mut c_void
    // }
    
    struct SimpleRustDescriptor {
        ndim: u8
    }

    unsafe extern "C++" {
        include!("ducc0/bindings/array_descriptor.h");

        // type ArrayDescriptor;
        // fn getSimpleDescriptor(ndim: u8) -> UniquePtr<SimpleDescriptor>;
        // type SimpleDescriptor;
        
        fn get_ndim(inp: &SimpleRustDescriptor) -> u8;
        fn set_ndim(desc: &mut SimpleRustDescriptor, n: u8);
    }
}

fn main() {
    // let a = Array::random((5, 6, 7), Uniform::new(0., 10.));
    // let slice = a.slice(s![.., 1, ..]);
    // println!("Whole array");
    // println!("{:8.4}", a);
    // println!("Slice");
    // println!("{:8.4}", slice);

    let mut arr = ffi::SimpleRustDescriptor{ndim: 2};
    println!("{}", ffi::get_ndim(&arr));

    println!("Before {}", ffi::get_ndim(&arr));
    ffi::set_ndim(&mut arr, 3);
    println!("After {}", ffi::get_ndim(&arr));

    // let b = ffi::getSimpleDescriptor(2: u8);
}
