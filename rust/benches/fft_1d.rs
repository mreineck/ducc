use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ducc0::fft_c2c;

use ndarray::Array;
use num_complex::Complex;

pub fn criterion_benchmark(z: &mut Criterion) {
    let shape = (500, 500, 10);
    let b = Array::from_elem(shape, Complex::<f64>::new(12., 0.));
    let mut c = Array::from_elem(shape, Complex::<f64>::new(0., 0.));
    let axes = vec![0, 1];
    z.bench_function("fft_c2c", |z| z.iter(|| fft_c2c(black_box(b.view()),
                                                      black_box(c.view_mut()),
                                                      black_box(&axes),
                                                      black_box(true),
                                                      black_box(1.),
                                                      black_box(1))));

        // fft_c2c(b.view(), c.view_mut(), &axes, true, 1., 1);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
