use std::path::PathBuf;

use clap::Parser;
use regalloc2::{
    checker::Checker, serialize::SerializableFunction, Algorithm, Block, Edit, Function,
    InstOrEdit, Output, RegallocOptions,
};

#[derive(Parser)]
/// Tool for testing regalloc2.
struct Args {
    /// Print the input function and the result of register allocation.
    #[clap(short = 'v')]
    verbose: bool,

    /// Input file containing a bincode-encoded SerializedFunction.
    input: PathBuf,

    /// Which register allocation algorithm to use.
    algorithm: CliAlgorithm,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum CliAlgorithm {
    Ion,
    Fastalloc,
}

impl From<CliAlgorithm> for Algorithm {
    fn from(cli_algo: CliAlgorithm) -> Algorithm {
        match cli_algo {
            CliAlgorithm::Ion => Algorithm::Ion,
            CliAlgorithm::Fastalloc => Algorithm::Fastalloc,
        }
    }
}

fn main() {
    pretty_env_logger::init();
    let args = Args::parse();

    let input = std::fs::read(&args.input).expect("could not read input file");
    let function: SerializableFunction =
        bincode::deserialize(&input).expect("could not deserialize input file");

    if args.verbose {
        println!("Input function: {function:?}");
    }

    let options = RegallocOptions {
        verbose_log: true,
        validate_ssa: true,
        algorithm: args.algorithm.into(),
    };
    let output = match regalloc2::run(&function, function.machine_env(), &options) {
        Ok(output) => output,
        Err(e) => {
            panic!("Register allocation failed: {e:#?}");
        }
    };

    if args.verbose {
        print_output(&function, &output);
    }

    let mut checker = Checker::new(&function, function.machine_env());
    checker.prepare(&output);
    if let Err(e) = checker.run() {
        panic!("Regsiter allocation checker failed: {e:#?}");
    }
}

fn print_output(func: &SerializableFunction, output: &Output) {
    print!("Register allocation result: {{\n");
    for i in 0..func.num_blocks() {
        let block = Block::new(i);
        let succs = func
            .block_succs(block)
            .iter()
            .map(|b| b.index())
            .collect::<Vec<_>>();
        let preds = func
            .block_preds(block)
            .iter()
            .map(|b| b.index())
            .collect::<Vec<_>>();
        print!("  block{}: # succs:{:?} preds:{:?}\n", i, succs, preds);
        for inst_or_edit in output.block_insts_and_edits(func, block) {
            match inst_or_edit {
                InstOrEdit::Inst(inst) => {
                    let op = if func.is_ret(inst) {
                        "ret"
                    } else if func.is_branch(inst) {
                        "branch"
                    } else {
                        "op"
                    };
                    let ops: Vec<_> = func
                        .inst_operands(inst)
                        .iter()
                        .zip(output.inst_allocs(inst))
                        .map(|(op, alloc)| format!("{op} => {alloc}"))
                        .collect();
                    let ops = ops.join(", ");
                    print!("    inst{}: {op} {ops}\n", inst.index(),);
                }
                InstOrEdit::Edit(Edit::Move { from, to }) => {
                    print!("    edit: move {to} <- {from}\n");
                }
            }
        }
    }
    print!("}}\n");
}
