use ndarray::Array1;
use clap::{Arg, App};

use std::time::Instant;
use std::fs::File;
use std::io::Write;

extern crate openblas_src;

mod utils;
mod viterbi_solver;
mod hmm;

use hmm::hmm::HMM;
use viterbi_solver::opti::GlobalOpti;
use viterbi_solver::dp::DPSolver;
use viterbi_solver::utils::SuperSequence;
use viterbi_solver::viterbi::decode;


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

fn global_opti<const D: usize>(hmm: &HMM<D>, sequence: &mut SuperSequence<D>, tags: &Vec<Vec<usize>>) {
    sequence.reorder();
    let mut model = GlobalOpti::new(hmm, sequence);
    model.build_model();
    let time = model.solve();
    let predictions = model.get_solutions();
    let solution = sequence.parse_solution(predictions);
    let error_rate = error_rate(&solution, tags);
    println!("Error rate is {:5} in {} secs", error_rate, time);
}

fn global_opti_exp<const D: usize>(hmm: &HMM<D>, sequence: &mut SuperSequence<D>, tags: &Vec<Vec<usize>>, config: &utils::Config) {
    let nb_repeat = 10;
    let mut output = File::create(config.output_path()).unwrap();
    for i in 0..nb_repeat {
        sequence.recompute_constraints(config.get_prop());
        let mut model = GlobalOpti::new(hmm, sequence);
        model.build_model();
        println!("config {:.2} {}/{}", config.get_prop(), i+1, nb_repeat);
        let runtime = model.solve();
        let predictions = model.get_solutions();
        let solution = sequence.parse_solution(predictions);
        let error_rate = error_rate(&solution, tags);
        let s = format!("{:.5} {}\n", error_rate, runtime);
        output.write(s.as_bytes()).unwrap();
    }
}

fn dp<'a, const D: usize>(hmm: &'a HMM<D>, sequence: &'a mut SuperSequence<'a, D>, tags: &Vec<Vec<usize>>) {
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

fn dp_exp<'a, const D: usize>(hmm: &'a HMM<D>, sequence: &'a mut SuperSequence<'a, D>, tags: &Vec<Vec<usize>>, config: &utils::Config) {
    let nb_repeat = 10;
    let mut output = File::create(config.output_path()).unwrap();
    let mut solver = DPSolver::new(hmm, sequence);
    for i in 0..nb_repeat {
        println!("config {:.2} {}/{}", config.get_prop(), i+1, nb_repeat);
        solver.refresh_constraints(config.get_prop());
        let start = Instant::now();
        solver.dp_solving();
        let runtime = start.elapsed().as_secs();
        let solution = solver.parse_solution();
        let error_rate = error_rate(&solution, tags);
        let s = format!("{:.5} {}\n", error_rate, runtime);
        output.write(s.as_bytes()).unwrap();
    }
}

fn error_rate(predictions: &Array1<Array1<usize>>, truth: &Vec<Vec<usize>>) -> f64 {
    let mut errors = 0.0;
    let mut total = 0.0;
    let mut error_per_seq = 0.0;
    let mut count = 0.0;
    for i in 0..predictions.len() {
        let mut local_err = 0.0;
        let prediction = &predictions[i];
        for j in 0..prediction.len() {
            if prediction[j] != truth[i][j] {
                errors += 1.0;
                local_err += 1.0;
            }
            total += 1.0;
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
        .about("Multiple viterbi with consistency constraints")
        .arg(Arg::new("config")
            .short('c')
            .long("config")
            .value_name("FILE")
            .about("configuration file")
            .takes_value(true)
            .required(true))
        .arg(Arg::new("exp")
            .short('e')
            .long("exp")
            .value_name("EXPERIENCES")
            .about("launch experiences or not")
            .takes_value(false))
        .get_matches();

    let mut config = if let Some(f) = matches.value_of("config") {
        utils::Config::from_config_file(std::path::PathBuf::from(f))
    } else {
        panic!("No config file provided")
    };

    let exp = matches.occurrences_of("exp") == 1;

    let sequences = config.get_sequences::<1>();
    let tags = config.get_tags();

    let mut info: Vec<Vec<Option<usize>>> = Vec::new();
    for ts in &tags {
        let mut c: Vec<Option<usize>> = Vec::new();
        for t in ts {
            c.push(Some(*t));
        }
        info.push(c);
    }

    let mut hmm = HMM::new(config.nstates, [config.nobs]);
    //hmm.train_supervised(&sequences, &tags, config.nstates);
    hmm.train_semi_supervised(&sequences, &info, config.nstates, 100, 0.01);
    
    let mut constraints = config.get_constraints();

    constraints.keep_prop(config.get_prop());

    let mut super_seq = SuperSequence::from(&sequences, &mut constraints, &hmm);
    super_seq.recompute_constraints(config.get_prop());

    if config.is_global_opti() {
        if exp {
            global_opti_exp(&hmm, &mut super_seq, &tags, &config);
        } else {
            global_opti(&hmm, &mut super_seq, &tags);
            //global_opti(&hmm, &mut super_seq, &truth);
        }
    } else if config.is_dp() {
        if exp {
            dp_exp(&hmm, &mut super_seq, &tags, &config);
        } else {
            dp(&hmm, &mut super_seq, &tags);
        }
    } else if config.is_viterbi() {
        //viterbi(&hmm, &sequences, &tags);
    }
}
