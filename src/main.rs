use ndarray::Array1;
use clap::{Arg, App};

use std::time::Instant;
use std::path;

mod utils;
mod viterbi_solver;

use viterbi_solver::viterbi;
use viterbi_solver::lagrangian::Lagrangian;
use viterbi_solver::hmm::HMM;
use viterbi_solver::constraints::Constraints;
use viterbi_solver::opti::GlobalOpti;


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

fn global_opti_exp(hmm: &HMM, sequences: &Array1<Array1<usize>>, constraints: &Constraints, tags: &Array1<Array1<usize>>, output_path: &path::PathBuf) {
    let props = Array1::range(0.0, 1.0, 0.05);
    let mut model = GlobalOpti::new(hmm, sequences, constraints);
    model.build_model();
    let mut error_rates: Array1<f64> = Array1::zeros(props.len());
    let mut runtime: Array1<f64> = Array1::zeros(props.len());
    let nb_repeat = 10;
    for i in 0..props.len() {
        let prop = props[i];
        println!("{}", prop);
        let mut sum_errors = 0.0;
        let mut sum_time = 0;
        let nb_repeat = 10;
        for _ in 0..nb_repeat {
            let time = model.solve(prop);
            sum_time += time;
            let predictions = model.get_solutions();
            let error_rate = error_rate(&predictions, tags);
            sum_errors += error_rate;
        }
        error_rates[i] = sum_errors / (nb_repeat as f64);
        runtime[i] = (sum_time as f64) / (nb_repeat as f64);
    }

    utils::write_metrics(&props, &error_rates, &runtime, &output_path);
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
    let matches = App::new("Consistent viterbi")
        .version("0.1")
        .author("Alexandre Dubray <alexandre.dubray@uclouvain.be")
        .about("Multiple viterbi with consistency constraints")
        .arg(Arg::new("config")
            .short('c')
            .long("config")
            .value_name("FILE")
            .about("configuration file")
            .takes_value(true)
            .required(true))
        .get_matches();

    let mut config = if let Some(f) = matches.value_of("config") {
        utils::Config::from_config_file(std::path::PathBuf::from(f))
    } else {
        panic!("No config file provided")
    };

    let a = config.get_transmatrix();
    let b = config.get_emissionmatrix();
    let pi = config.get_initprob();
    let hmm = HMM::new(a, b, pi);

    let sequences = config.get_sequences();
    let tags = config.get_tags();
    let constraints = config.get_constraints();

    let output_path = config.output_path();
    if config.is_global_opti() {
        global_opti_exp(&hmm, &sequences, &constraints, &tags, output_path);
    } else if config.is_viterbi() {
    }
}
