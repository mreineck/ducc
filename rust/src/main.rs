#[cxx::bridge(namespace = "ducc0")]
mod ffi {
    unsafe extern "C++" {
        include!("ducc0/example_header.h");
        fn cxxsquare(x: f64) -> f64;
    }
}

fn main() {
    let a = ffi::cxxsquare(5.0);
    println!("{}", a);
}
