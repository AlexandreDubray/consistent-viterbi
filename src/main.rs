use ndarray::Array1;
use clap::{Arg, App};

use std::time::Instant;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use rand::prelude::*;
use ndarray::Array2;

extern crate openblas_src;

mod utils;
mod viterbi_solver;
mod hmm;

use hmm::hmm::HMM;
use viterbi_solver::opti::GlobalOpti;
use viterbi_solver::dp::DPSolver;
use viterbi_solver::utils::SuperSequence;
use viterbi_solver::viterbi::decode;
use viterbi_solver::constraints::Constraints;


/*
fn viterbi<const D: usize>(hmm: &HMM<D>, sequences: &Vec<Vec<[usize; D]>>, tags: &Array1<Array1<usize>>) {
    let mut max_seq_size = 0;
    for sequence in sequences {
        max_seq_size = max_seq_size.max(sequence.len());
    }

    let start = Instant::now();
    let predictions = sequences.map(|sequence| -> Array1<usize> { decode(&sequence, hmm) });
    let elapsed = start.elapsed().as_secs();
    let error_rate = error_rate(&predictions, tags);
    println!("Error rate {:.5} in {} sec", error_rate, elapsed);
}
*/

fn global_opti<const D: usize>(hmm: &HMM<D>, sequence: &mut SuperSequence<D>, tags: &Vec<Vec<Option<usize>>>) {
    sequence.reorder();
    let mut model = GlobalOpti::new(hmm, sequence);
    model.build_model();
    let time = model.solve();
    let predictions = model.get_solutions();
    let solution = sequence.parse_solution(predictions);
    let error_rate = error_rate(&solution, tags);
    println!("Error rate is {:5} in {} secs", error_rate, time);
}

fn dp<'a, const D: usize>(hmm: &'a HMM<D>, sequence: &'a mut SuperSequence<'a, D>, tags: &Vec<Vec<Option<usize>>>) {
    let mut solver = DPSolver::new(hmm, sequence);
    solver.reorder();
    let start = Instant::now();
    //solver.dp_solving();
    solver.iterative_solving();
    let elapsed = start.elapsed().as_millis();
    let solution = solver.parse_solution();
    let error_rate = error_rate(&solution, tags);
    println!("Error rate is {:5} in {} secs", error_rate, elapsed);
}

fn error_rate(predictions: &Array1<Array1<usize>>, truth: &Vec<Vec<Option<usize>>>) -> f64 {
    let mut errors = 0.0;
    let mut total = 0.0;
    let mut error_per_seq = 0.0;
    let mut count = 0.0;
    for i in 0..predictions.len() {
        let mut local_err = 0.0;
        let prediction = &predictions[i];
        for j in 0..prediction.len() {
            match truth[i][j] {
                Some(t) => {
                    if prediction[j] != t {
                        errors += 1.0;
                        local_err += 1.0;
                    }
                    total += 1.0;
                },
                None => ()
            }
        }
        if local_err != 0.0 {
            error_per_seq += local_err;
            count += 1.0;
        }
    }
    println!("Average error per bad sequence {}", error_per_seq/count);
    errors/total
}

fn main() {
    let matches = App::new("Consistent viterbi")
        .version("0.1")
        .author("Alexandre Dubray <alexandre.dubray@uclouvain.be>")
        .about("Hidden Markov Model with consistency constraints")
        .arg(Arg::new("INPUT")
            .short('i')
            .long("input")
            .about("Path to the file with the input sequences")
            .takes_value(true)
            .required(true))
        .arg(Arg::new("TAGS")
            .short('t')
            .long("tags")
            .about("Path to the file containing the tags of the sequences")
            .takes_value(true)
            .required(true))
        .arg(Arg::new("CONTROL")
            .long("control")
            .about("Path to control tags")
            .takes_value(true)
            .required(true))
        .arg(Arg::new("OUTPUT")
            .short('o')
            .long("output")
            .about("Path for output")
            .takes_value(true)
            .default_value("."))
        .arg(Arg::new("CONSTRAINT")
            .short('c')
            .long("constraint")
            .about("Path to the file containing the constraints")
            .takes_value(true)
            .required(true))
        .arg(Arg::new("NSTATES")
            .short('n')
            .long("nstates")
            .about("Number of hidden states in the HMM")
            .takes_value(true)
            .required(true))
        .arg(Arg::new("NOBS")
            .short('b')
            .long("nobs")
            .about("Number of observations per features")
            .takes_value(true)
            .min_values(1))
        .arg(Arg::new("PROP")
            .short('p')
            .long("prop")
            .about("Propostion of constraints to include when decoding")
            .required(true)
            .takes_value(true))
        .get_matches();

    let input_path = PathBuf::from(matches.value_of("INPUT").unwrap());
    let tags_path = PathBuf::from(matches.value_of("TAGS").unwrap());
    let control_tags_path = PathBuf::from(matches.value_of("CONTROL").unwrap());
    let output_path = PathBuf::from(matches.value_of("OUTPUT").unwrap());
    let cstr_path = PathBuf::from(matches.value_of("CONSTRAINT").unwrap());
    let nstates = matches.value_of("NSTATES").unwrap().parse::<usize>().unwrap();
    let nobs: Vec<usize> = matches.values_of("NOBS").unwrap().map(|x| x.parse::<usize>().unwrap()).collect();
    let prop = matches.value_of("PROP").unwrap().parse::<f64>().unwrap();

    let sequences = utils::load_sequences::<2>(&input_path);
    let tags = utils::load_tags(&tags_path);
    let control_tags = utils::load_tags(&control_tags_path);
    //let mut constraints = Constraints::from_file(&cstr_path);
    let mut constraints = Constraints::from_tags(&tags, 4);
    constraints.keep_prop(prop);

    let mut hmm = HMM::new(nstates, [nobs[0], nobs[1]]);
    //hmm.train_supervised(&sequences, &tags);
    let mut rng = thread_rng();
    let mut a_prio = Array2::from_shape_fn((nstates, nstates), |_| rng.gen::<f64>());
    a_prio[[1, 2]] = 0.0;
    a_prio[[2, 1]] = 0.0;
    a_prio[[1, 3]] = 0.0;
    a_prio[[3, 1]] = 0.0;
    a_prio[[2, 3]] = 0.0;
    a_prio[[3, 2]] = 0.0;
    for state in 0..nstates {
        let mut a_row = a_prio.row_mut(state);
        let s = a_row.sum();
        a_row /= s;
    }
    hmm.train_semi_supervised(&sequences, &tags, Some(a_prio), 100, 0.01);

    let mut super_seq = SuperSequence::from(&sequences, &mut constraints, &hmm);
    super_seq.recompute_constraints(prop);
    global_opti(&hmm, &mut super_seq, &control_tags);
}
