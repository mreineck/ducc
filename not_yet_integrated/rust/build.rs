fn main() {
    cc::Build::new()
        .cpp(true)
        .file("ducc_rust.cc")
        .flag_if_supported("-std=c++20")
        .include("cpp_src")
        .compile("ducc0");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=ducc_rust.cc");
}
