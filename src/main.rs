use ndarray::Array1;

use std::time::Instant;
use std::env;

mod utils;
mod viterbi_solver;
mod lp_exp;

use viterbi_solver::viterbi;
use viterbi_solver::lagrangian::Lagrangian;
use viterbi_solver::hmm::HMM;
use viterbi_solver::constraints::Constraints;
use viterbi_solver::opti::GlobalOpti;

use structopt::StructOpt;

#[derive(StructOpt)]
struct Cli {
    method: String,
    hmm_path: std::path::PathBuf,
    input_path: std::path::PathBuf,
    nstates: usize,
    nobs: usize,
    prop_consistency_cstr: f64
}


fn log(n: &f64) -> f64 {
    if *n == 0.0 {
        return f64::NEG_INFINITY;
    } else {
        return n.log(10.0);
    }
}

fn viterbi(hmm: HMM, sequences: &Array1<Array1<usize>>) -> Array1<Array1<usize>> {
    let mut max_seq_size = 0;
    for sequence in sequences {
        max_seq_size = max_seq_size.max(sequence.len());
    }

    let mut viterbi = viterbi::Viterbi::new(&hmm, max_seq_size);
    let start = Instant::now();
    println!("Start of the predictions");
    let predictions = sequences.map(|sequence| -> Array1<usize> { viterbi.solve(&sequence) });
    let elapsed = start.elapsed().as_secs();
    println!("Viterbi solved in {} seconds", elapsed);
    predictions
}

fn lagrangian(hmm: HMM, sequences: Vec<Array1<usize>>, constraints: Constraints) {
    let mut max_seq_size = 0;
    for sequence in &sequences {
        max_seq_size = max_seq_size.max(sequence.len());
    }

    println!("Creating lagrangian");
    let mut lagrangian = Lagrangian::new(hmm, sequences, max_seq_size, constraints);
    println!("Solving lagrangian");
    lagrangian.solve();
}

fn global_opti(hmm: &HMM, sequences: &Array1<Array1<usize>>, constraints: &Constraints, prop_consistency_cstr: f64) -> Array1<Array1<usize>> {
    let mut model = GlobalOpti::new(hmm, sequences, constraints);
    model.build_model();
    model.solve(prop_consistency_cstr);
    let predictions = model.get_solutions();
    predictions
}

fn error_rate(predictions: &Array1<Array1<usize>>, truth: &Array1<Array1<usize>>) -> f64 {
    let mut errors = 0.0;
    let mut total = 0.0;
    for i in 0..predictions.len() {
        let prediction = &predictions[i];
        for j in 0..prediction.len() {
            if prediction[j] != truth[i][j] {
                errors += 1.0;
            }
            total += 1.0;
        }
    }
    errors/total
}

fn main() {
    //lp_exp::launch_exp();
    let mut args = Cli::from_args();

    println!("Loading matrices");
    args.hmm_path.push("b");
    let emissionprob = utils::read_matrix(&args.hmm_path, args.nstates, args.nobs).map(log);
    args.hmm_path.set_file_name("A");
    let transmat = utils::read_matrix(&args.hmm_path, args.nstates, args.nstates).map(log);
    args.hmm_path.set_file_name("pi");
    let pi = utils::read_matrix(&args.hmm_path, 1, args.nstates).row(0).map(log);
    let hmm = HMM::new(transmat, emissionprob, pi);
    println!("Loading sequences");
    args.input_path.push("sentences");
    let sequences = utils::load_sequences(&args.input_path);
    args.input_path.set_file_name("tags");
    let tags = utils::load_sequences(&args.input_path);

    println!("Loading constraints");
    args.input_path.set_file_name("constraints");
    let constraints = Constraints::from_file(&args.input_path);
    
    let predictions = {
        if args.method == "viterbi" {
            viterbi(hmm, &sequences)
        } else if args.method == "global_opti" {
            global_opti(&hmm, &sequences, &constraints, args.prop_consistency_cstr)
        } else {
            panic!("Unknown solving method: {}", args.method)
        }
    };
    let e = error_rate(&predictions, &tags);
    println!("Error rate {}", e);
    args.hmm_path.set_file_name(args.method + "_output");
    utils::write_outputs(&args.hmm_path, &predictions);
}
