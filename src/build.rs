use cxx_build::CFG;

fn main() {
    CFG.include_prefix = "";
    cxx_build::bridge("src/main.rs")
    //.file("ducc.cc")
    .compile("ducc-demo");
}
