use ndarray::Array1;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use super::viterbi_solver::constraints::Constraints;

pub fn load_sequences(path: &PathBuf) -> Array1<Array1<usize>> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);

    let mut ret: Vec<Array1<usize>> = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let sequence = line.split(" ").map(|x| x.parse::<usize>().unwrap()).collect();
        ret.push(sequence)
    }
    Array1::from_vec(ret)
}

pub struct Config {
    method: String,
    hmm_path: PathBuf,
    input_path: PathBuf,
    output_path: PathBuf,
    pub nstates: usize,
    pub nobs: usize,
    prop_constraints: f64
}

impl Config {

    fn new() -> Self {
        let method = String::from("viterbi");
        let hmm_path = PathBuf::from(".");
        let input_path = PathBuf::from(".");
        let output_path = PathBuf::from(".");
        let nstates = 0;
        let nobs = 0;
        let prop_constraints = 0.0;

        Self {method, hmm_path, input_path, output_path, nstates, nobs, prop_constraints}
    }

    pub fn from_config_file(filename: PathBuf) -> Self {
        let mut instance = Config::new();
        let file = match File::open(filename) {
            Ok(f) => f,
            Err(error) => panic!("Can not open file config file: {:?}", error)
        };

        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line.unwrap();
            let splits: Vec<&str> = line.split("=").collect();
            if splits.len() != 2 {
                panic!("Wrong line in config file");
            }
            let option = splits[0];
            let value = String::from(splits[1]);
            match option {
                "method" => instance.method = value,
                "hmm_path" => instance.hmm_path = PathBuf::from(value),
                "input_path" => instance.input_path = PathBuf::from(value),
                "output_path" => instance.output_path = PathBuf::from(value),
                "nstates" => instance.nstates = value.parse().unwrap(),
                "nobs" => instance.nobs = value.parse().unwrap(),
                "prop" => instance.prop_constraints = value.parse().unwrap(),
                _ => panic!("Unknown option in config file: {:?}", option)
            };
        }
        instance.hmm_path.push("tmp");
        instance.input_path.push("tmp");
        instance.output_path.push(format!("{:.2}", instance.prop_constraints));
        instance
    }


    pub fn get_sequences(&mut self) -> Array1<Array1<usize>> {
        self.input_path.set_file_name("sequences");
        load_sequences(&self.input_path)
    }

    pub fn get_tags(&mut self) -> Array1<Array1<usize>> {
        self.input_path.set_file_name("tags");
        load_sequences(&self.input_path)
    }

    pub fn get_constraints(&mut self) -> Constraints {
        self.input_path.set_file_name("constraints");
        Constraints::from_file(&self.input_path)
    }

    pub fn is_global_opti(&self) -> bool {
        self.method == "global_opti"
    }

    pub fn is_viterbi(&self) -> bool {
        self.method == "viterbi"
    }

    pub fn is_dp(&self) -> bool {
        self.method == "dp"
    }

    pub fn output_path(&self) -> &PathBuf {
        &self.output_path
    }

    pub fn get_prop(&self) -> f64 {
        self.prop_constraints
    }
}

