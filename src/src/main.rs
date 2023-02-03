// use ndarray::Array;
// use ndarray_rand::RandomExt;
// use ndarray_rand::rand_distr::Uniform;
// 
// use num_complex::Complex;



#[cxx::bridge(namespace = "ducc0")]
mod ffi {
    unsafe extern "C++" {
        include!("ducc0/bindings/array_descriptor.h");
    }
}

fn main() {
    //CFG.exported_header_dirs.push("ducc0/ducc/src");
    //CFG.exported_header_dirs.push("./ducc/src");
    //CFG.exported_header_dirs.push("./ducc/src/ducc0");
    //CFG.exported_header_dirs.push("ducc0/ducc/src/ducc0");

    // let a = Array::random((2, 5), Uniform::new(0., 10.));
    // println!("{:8.4}", a);
}
