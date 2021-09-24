use ndarray::Array1;

use std::time::Instant;

mod utils;
mod viterbi_solver;
//mod opti;
//mod constraints;

use viterbi_solver::viterbi;
use viterbi_solver::lagrangian::Lagrangian;
use viterbi_solver::hmm::HMM;
use viterbi_solver::constraints::Constraints;
use viterbi_solver::opti;

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

    let mut viterbi = viterbi::Viterbi::new(hmm, max_seq_size);
    let start = Instant::now();
    println!("Start of the predictions");
    let predictions = sequences.map(|sequence| -> Array1<usize> { viterbi.solve(&sequence) });
    let elapsed = start.elapsed().as_secs();
    println!("Viterbi solved in {} seconds", elapsed);
    utils::write_outputs("predictions_viterbi.out", &predictions);
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

    let nstates = 472;
    let nobs = 56057;

    println!("Loading matrices");
    let transmat = utils::read_matrix("data/transmat", nstates, nstates).map(log);
    let emissionprob = utils::read_matrix("data/emissionmat", nstates, nobs).map(log);
    let pi = utils::read_matrix("data/pi", 1, nstates).row(0).map(log);
    let hmm = HMM::new(transmat, emissionprob, pi);
    println!("Loading sequences");
    let sequences = utils::load_sequences("data/sentences_reduced.in");
    let tags = utils::load_sequences("data/tags_reduced.in");
    let predictions = viterbi(hmm, &sequences);
    let e = error_rate(&predictions, &tags);
    println!("Error rate {}", e);

    //println!("Loading constraints");
    //let constraints = Constraints::from_file("data/constraints.in");

    //lagrangian(hmm, sequences, constraints);
    //let model = opti::create_model(hmm, sequences);
}
