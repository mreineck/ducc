fn main() {
    cc::Build::new()
        .cpp(true)
        .file("ducc_rust.cc")
        .include(".")
        .compile("ducc0");

    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=ducc_rust.cc");
}
