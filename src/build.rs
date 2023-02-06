fn main() {
    cc::Build::new()
        .cpp(true)
        .file("ducc_rust.cc")
        .include(".")
        .compile("libducc0.a");
}
