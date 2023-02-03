use ndarray::{Array, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

// use num_complex::Complex;


// static constexpr size_t maxdim=10;
//
// array<uint64_t, maxdim> shape;
// array<int64_t, maxdim> stride;
//
// void *data;
// uint8_t ndim;
// uint8_t dtype;

#[cxx::bridge(namespace = "ducc0")]
mod ffi {
    unsafe extern "C++" {
        include!("ducc0/bindings/array_descriptor.h");

        // type ArrayDescriptor;

        // fn printArrayDescriptor(inp: ArrayDescriptor) -> ;
    }
}

fn main() {
    let a = Array::random((5, 6, 7), Uniform::new(0., 10.));
    let slice = a.slice(s![.., 1, ..]);
    println!("Whole array");
    println!("{:8.4}", a);
    println!("Slice");
    println!("{:8.4}", slice);
}
