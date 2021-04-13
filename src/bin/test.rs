use arbitrary::{Arbitrary, Unstructured};
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

fn main() {
    const SIZE: usize = 1000 * 1000;
    env_logger::init();
    let env = machine_env();
    for iter in 0..3 {
        let func = create_random_func(iter, SIZE);
        eprintln!("==== {} instructions", func.insts());
        let mut stats: ion::Stats = ion::Stats::default();
        for i in 0..1000 {
            let out = ion::run(&func, &env).expect("regalloc did not succeed");
            if i == 0 {
                stats = out.stats;
            }
        }
        eprintln!("Stats: {:?}", stats);
    }
}
