use clap::{Arg, App};

use std::time::Instant;

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

extern crate openblas_src;

mod utils;
mod viterbi_solver;
mod hmm;

use hmm::hmm::HMM;
use viterbi_solver::opti::IPSolver;
use viterbi_solver::dp::DPSolver;
use viterbi_solver::utils::SuperSequence;
use viterbi_solver::constraints::Constraints;
use viterbi_solver::cp::CPSolver;
use viterbi_solver::Solver;
use viterbi_solver::cfn::write_cfn;


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
        .arg(Arg::new("OUTPUT")
            .short('o')
            .long("output")
            .about("Path for output")
            .takes_value(true)
            .default_value("."))
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
        .arg(Arg::new("TRAIN")
            .short('t')
            .long("train")
            .about("If present, the HMM is learned from the data")
            .takes_value(false))
        .arg(Arg::new("SUPERVISED")
            .short('s')
            .long("supervised")
            .about("If the hmm is learned, this flag indicates to use the supervised learning process"))
        .get_matches();

    let mut input_path = PathBuf::from(matches.value_of("INPUT").unwrap());
    input_path.push("tmp");
    let mut output_path = PathBuf::from(matches.value_of("OUTPUT").unwrap());
    output_path.push("tmp");
    let nstates = matches.value_of("NSTATES").unwrap().parse::<usize>().unwrap();
    let nobs: Vec<usize> = matches.values_of("NOBS").unwrap().map(|x| x.parse::<usize>().unwrap()).collect();
    let prop = matches.value_of("PROP").unwrap().parse::<f64>().unwrap();


    println!("Loading data");
    input_path.set_file_name("sequences");
    let sequences = utils::load_sequences::<2>(&input_path);
    input_path.set_file_name("tags");
    let tags = utils::load_tags(&input_path);
    input_path.set_file_name("test_tags");
    let control_tags = utils::load_tags(&input_path);
    let mut constraints = Constraints::from_tags(&control_tags);

    let hmm = match matches.is_present("TRAIN") {
        true => {
            let mut h = HMM::new(nstates, [nobs[0], nobs[1]]);
            if matches.is_present("SUPERVISED") {
                h.maximum_likelihood_estimation(&sequences, &tags);
            } else {
                h.train(&sequences, &tags, 1000, 0.001);
            }
            h.write(&mut input_path);
            h
        },
        false => {
            input_path.set_file_name("hmm.json");
            HMM::from_json(&input_path)
        }
    };

    let mut super_seq = SuperSequence::from(&sequences, &mut constraints, &hmm);
    super_seq.recompute_constraints(prop);
    let run_cfn = false;
    let nb_run = 1;
    for run in 0..nb_run {
        output_path.set_file_name(format!("{}_{}", prop, run));
        let mut file = File::create(&output_path).unwrap();
        if prop != 0.0 && prop != 1.0 {
            super_seq.recompute_constraints(prop);
        }
        if run_cfn {
            let compile_time = write_cfn(&hmm, &super_seq, &mut output_path, format!("problem_{}_{}.cfn", prop, run));
            file.write_all(format!("{}\n", compile_time).as_bytes()).unwrap();
        } else {
            let mut solver = CPSolver::new(&hmm, &super_seq);
            //let mut solver = IPSolver::new(&hmm, &super_seq);
            //let mut solver = DPSolver::new(&hmm, &super_seq);
            println!("[{} EXP {}] Run {}/{}", solver.get_name(),  prop, run+1, nb_run);
            let start = Instant::now();
            solver.solve();
            let elapsed = start.elapsed().as_millis();

            //file.write_all(format!("{}\n{}\n", solver.get_objective(), elapsed).as_bytes()).unwrap();
            file.write_all(format!("{} {}\n{}\n", solver.get_objective(), solver.get_explored_nodes(), elapsed).as_bytes()).unwrap();
            let solution = solver.get_solution();
            for i in 0..solution.len() {
                file.write_all(format!("{} {}\n", super_seq[i].seq, solution[i]).as_bytes()).unwrap();
            }
        }
    }
}
