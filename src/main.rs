use ndarray::Array1;
use clap::{Arg, App};

use std::time::Instant;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use rand::prelude::*;

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
use viterbi_solver::cp::CPSolver;


fn global_opti<const D: usize>(hmm: &HMM<D>, sequence: &mut SuperSequence<D>, tags: &Vec<Vec<Option<usize>>>) -> (Array1<Array1<usize>>, f64) {
    sequence.reorder();
    let mut model = GlobalOpti::new(hmm, sequence);
    model.build_model();
    let time = model.solve();
    let predictions = model.get_solutions();
    let solution = sequence.parse_solution(predictions);
    let error_rate = error_rate(&solution, tags);
    println!("Error rate is {:5} in {} secs", error_rate, time);
    (solution, model.objective)
}

fn dp<'a, const D: usize>(hmm: &'a HMM<D>, sequence: &'a mut SuperSequence<'a, D>, tags: &Vec<Vec<Option<usize>>>) -> (Array1<Array1<usize>>, f64) {
    let mut solver = DPSolver::new(hmm, sequence);
    solver.reorder();
    let start = Instant::now();
    solver.dp_solving();
    let elapsed = start.elapsed().as_millis();
    let solution = solver.parse_solution();
    let error_rate = error_rate(&solution, tags);
    println!("Error rate is {:5} in {} secs", error_rate, elapsed);
    (solution, solver.objective)
}

fn cp<'a, const D: usize>(hmm: &'a HMM<D>, sequence: &'a SuperSequence<'a, D>, tags: &Vec<Vec<Option<usize>>>) -> (Array1<Array1<usize>>, f64) {
    let mut constraints: Vec<Vec<usize>> = Vec::new();
    let mut m: Array1<i32> = Array1::from_elem(sequence.nb_cstr, -1);
    for t in 0..sequence.len() {
        if sequence[t].is_constrained() {
            let ucomp = sequence[t].constraint_component as usize;
            if m[ucomp] == -1 {
                m[ucomp] = constraints.len() as i32;
                constraints.push(Vec::new());
            }
            constraints[m[ucomp] as usize].push(t);
        }
    }
    println!("Number of constrained: {}", constraints.len());
    let mut solver = CPSolver::new(hmm, sequence, &constraints);
    let start = Instant::now();
    solver.solve();
    let elapsed = start.elapsed().as_secs();
    print!("CP solver solved in {} seconds", elapsed);
    let solution = sequence.parse_solution(solver.get_solution());
    let objective = solver.get_objective();
    (solution, objective)

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
    let mut output_path = PathBuf::from(matches.value_of("OUTPUT").unwrap());
    let cstr_path = PathBuf::from(matches.value_of("CONSTRAINT").unwrap());
    let nstates = matches.value_of("NSTATES").unwrap().parse::<usize>().unwrap();
    let nobs: Vec<usize> = matches.values_of("NOBS").unwrap().map(|x| x.parse::<usize>().unwrap()).collect();
    let prop = matches.value_of("PROP").unwrap().parse::<f64>().unwrap();

    println!("Loading data");
    let sequences = utils::load_sequences::<2>(&input_path);
    let tags = utils::load_tags(&tags_path);
    let control_tags = utils::load_tags(&control_tags_path);
    //let mut constraints = Constraints::from_file(&cstr_path);
    let mut constraints = Constraints::from_tags(&control_tags, 4);
    constraints.keep_prop(prop);

    let hmm = HMM::from_json(&mut output_path);

    let mut super_seq = SuperSequence::from(&sequences, &mut constraints, &hmm);
    super_seq.reorder();
    super_seq.recompute_constraints(prop);
    //let (solution, objective) = global_opti(&hmm, &mut super_seq, &control_tags);
    //let (solution, objective) = dp(&hmm, &mut super_seq, &control_tags);
    let (solution, objective) = cp(&hmm, &mut super_seq, &control_tags);

    output_path.set_file_name(format!("solution_{}", prop));
    let mut file = File::create(&output_path).unwrap();
    file.write_all(format!("{}\n", objective).as_bytes()).unwrap();

    for i in 0..solution.len() {
        let sol = &solution[i];
        let exp = &control_tags[i];
        for j in 0..sol.len() {
            let predicted = sol[j];
            let expected = match exp[j] {
                Some(tag) => tag as i32,
                None => -1
            };
            file.write_all(format!("{} {}\n", predicted, expected).as_bytes()).unwrap();
        }
        file.write_all("\n".as_bytes()).unwrap();
    }
}
