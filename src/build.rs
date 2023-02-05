use cxx_build::CFG;

fn main() {
    CFG.include_prefix = "";
    cxx_build::bridge("src/main.rs")
    .file("ducc_rust.cc")
    .flag_if_supported("-std=c++17")
    .compile("ducc-demo");
}
