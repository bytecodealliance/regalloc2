//! Criterion-based benchmark target that computes insts/second for
//! arbitrary inputs.

use arbitrary::{Arbitrary, Unstructured};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use regalloc2::fuzzing::func::{machine_env, Func};
use regalloc2::ion;
use regalloc2::Function;

fn create_random_func(seed: u64, size: usize) -> Func {
    let mut bytes: Vec<u8> = vec![];
    bytes.resize(size, 0);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.fill(&mut bytes[..]);
    loop {
        let mut u = Unstructured::new(&bytes[..]);
        match Func::arbitrary(&mut u) {
            Ok(f) => {
                return f;
            }
            Err(arbitrary::Error::NotEnoughData) => {
                let len = bytes.len();
                bytes.resize(len + 1024, 0);
                rng.fill(&mut bytes[len..]);
            }
            Err(e) => panic!("unexpected error: {:?}", e),
        }
    }
}

fn run_regalloc(c: &mut Criterion) {
    const SIZE: usize = 1000 * 1000;
    env_logger::init();
    let env = machine_env();
    let mut group = c.benchmark_group("benches");
    for iter in 0..3 {
        let func = create_random_func(iter, SIZE);
        eprintln!("==== {} instructions", func.insts());
        group.throughput(Throughput::Elements(func.insts() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(iter), &iter, |b, _| {
            b.iter(|| {
                // For fair comparison with regalloc.rs, which needs
                // to clone its Func on every alloc, we clone
                // too. Seems to make a few percent difference.
                let func = func.clone();
                ion::run(&func, &env).expect("regalloc did not succeed");
            });
        });
    }
    group.finish();
}

criterion_group!(benches, run_regalloc);
criterion_main!(benches);
