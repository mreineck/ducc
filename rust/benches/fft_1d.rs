use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ducc0::fft_c2c;
use ndarray::{Array, ArrayBase, Data, DataMut, Dimension};
use ndrustfft::{ndfft, FftHandler, FftNum};
use num::traits::FloatConst;
use num_complex::Complex;

pub fn criterion_ducc(z: &mut Criterion) {
    let shape = (500, 500, 10);
    let b = Array::from_elem(shape, Complex::<f64>::new(12., 0.));
    let mut c = Array::from_elem(shape, Complex::<f64>::new(0., 0.));
    let axes = vec![0, 1];
    z.bench_function("ducc_c2c", |z| {
        z.iter(|| {
            fft_c2c(
                black_box(b.view()),
                black_box(c.view_mut()),
                black_box(&axes),
                black_box(true),
                black_box(1.),
                black_box(1),
            )
        })
    });

    z.bench_function("ndrustfft_c2c", |z| {
        z.iter(|| {
            rustfft_2d_c2c(
                black_box(&b),
                black_box(&mut c),
                //   black_box(&axes),
                //black_box(true),
                //black_box(1.),
                //black_box(1))
            )
        })
    });
}

fn rustfft_2d_c2c<R, S, T, D>(
    inp: &ArrayBase<R, D>,
    out: &mut ArrayBase<S, D>,
    //axes: &Vec<usize>,
    //forward: bool,
    //fct: f64,
    //nthreads: usize,
) where
    T: FftNum + FloatConst,
    R: Data<Elem = Complex<T>>,
    S: Data<Elem = Complex<T>> + DataMut,
    D: Dimension,
{
    let mut handler_ax0 = FftHandler::<T>::new(inp.shape()[0]);
    let mut handler_ax1 = FftHandler::<T>::new(inp.shape()[1]);
    {
        let mut work: Array<Complex<T>, D> = Array::zeros(inp.raw_dim());
        ndfft(&inp, &mut work, &mut handler_ax1, 1);
        ndfft(&work, out, &mut handler_ax0, 0);
    }
}

criterion_group!(benches, criterion_ducc);
criterion_main!(benches);
