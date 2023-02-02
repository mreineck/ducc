fn main() {
    cxx_build::bridge("src/main.rs")
    //.file("ducc.cc")
    .compile("ducc-demo");
}
