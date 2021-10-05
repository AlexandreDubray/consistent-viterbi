use csv::ReaderBuilder;
use ndarray::{Array2, Array1};
use ndarray_csv::Array2Reader;

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use super::viterbi_solver::constraints::Constraints;

fn log(n: &f64) -> f64 {
    if *n == 0.0 {
        return f64::NEG_INFINITY;
    } else {
        return n.log(10.0);
    }
}

pub fn read_matrix(path: &PathBuf, nrows: usize, ncols: usize) -> Array2<f64> {
    let file = File::open(path).unwrap();
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    reader.deserialize_array2((nrows, ncols)).unwrap()
}

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

fn array_to_str(a: &Array1<usize>) -> String {
    let mut s = String::new();
    s.push_str(&format!("{}", a[0]));
    for i in 1..a.len() {
        s.push(' ');
        s.push_str(&format!("{}", a[i]));
    }
    s
}

pub fn write_outputs(path: &PathBuf, outputs: &Array1<Array1<usize>>) {
    let mut file = File::create(path).unwrap();
    for output in outputs {
        file.write(array_to_str(output).as_bytes()).expect("Can not write");
        file.write("\n".as_bytes()).expect("Can not write");
    }
}

pub fn write_metrics(props: &Array1<f64>, errors: &Array1<f64>, runtimes: &Array1<f64>, path: &PathBuf) {
    let mut file = File::create(path).unwrap();
    for i in 0..props.len() {
        let s = format!("{} {} {}\n", props[i], errors[i], runtimes[i]);
        file.write(s.as_bytes());
    }
}

pub struct Config {
    method: String,
    hmm_path: PathBuf,
    input_path: PathBuf,
    nstates: usize,
    nobs: usize,
}

impl Config {

    fn new() -> Self {
        let method = String::from("viterbi");
        let hmm_path = PathBuf::from(".");
        let input_path = PathBuf::from(".");
        let nstates = 0;
        let nobs = 0;
        Self {method, hmm_path, input_path, nstates, nobs}
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
                "nstates" => instance.nstates = value.parse().unwrap(),
                "nobs" => instance.nobs = value.parse().unwrap(),
                _ => panic!("Unknown option in config file: {:?}", option)
            };
        }
        instance
    }

    pub fn get_transmatrix(&mut self) -> Array2<f64> {
        self.hmm_path.set_file_name("A");
        read_matrix(&self.hmm_path, self.nstates, self.nstates).map(log)
    }

    pub fn get_emissionmatrix(&mut self) -> Array2<f64> {
        self.hmm_path.set_file_name("b");
        read_matrix(&self.hmm_path, self.nstates, self.nobs).map(log)
    }

    pub fn get_initprob(&mut self) -> Array1<f64> {
        read_matrix(&self.hmm_path, 1, self.nstates).row(0).map(log)
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
}

