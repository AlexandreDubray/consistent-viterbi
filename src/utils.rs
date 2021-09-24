use csv::ReaderBuilder;
use ndarray::{Array2, Array1};
use ndarray_csv::Array2Reader;

use std::fs::File;
use std::io::{BufRead, BufReader, Write};

pub fn read_matrix(filename: &str, nrows: usize, ncols: usize) -> Array2<f64> {
    let file = match File::open(filename) {
        Ok(f) => f,
        Err(error) => panic!("Unable to open file {:?}. {:?}", filename, error),
    };
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    match reader.deserialize_array2((nrows, ncols)) {
        Ok(a) => a,
        Err(error) => panic!("Unable to parse matrix in file {:?}. {:?}", filename, error),
    }
}

pub fn load_sequences(filename: &str) -> Array1<Array1<usize>> {
    let file = File::open(filename).unwrap();
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

pub fn write_outputs(filename: &str, outputs: &Array1<Array1<usize>>) {
    let mut file = match File::create(filename) {
        Ok(f) => f,
        Err(error) => panic!("Can not create file {}. {:?}", filename, error),
    };
    for output in outputs {
        file.write(array_to_str(output).as_bytes()).expect("Can not write");
        file.write("\n".as_bytes()).expect("Can not write");
    }
}
