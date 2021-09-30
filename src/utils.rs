use csv::ReaderBuilder;
use ndarray::{Array2, Array1};
use ndarray_csv::Array2Reader;

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

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
