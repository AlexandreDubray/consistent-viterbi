use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::fs::File;
use ndarray::Array1;
use std::collections::HashSet;

struct ConsistencyConstraint {
    seq_id_from: usize,
    seq_id_to: usize,
    timestamp_from: usize,
    timestamp_to: usize
}

pub struct Constraints {
    pub components: Array1<Vec<(usize, usize)>>,
    pub constrained_elements: HashSet<(usize, usize)>
}

impl Constraints {

    fn parse_usize(s: &str) -> usize {
        match s.parse::<usize>() {
            Ok(v) => v,
            Err(error) => panic!("Can not parse {} as usize: {:?}", s, error)
        }
    }

    pub fn from_file(path: &PathBuf) -> Self {
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        // Each component is separated by an empty line
        let mut component: Vec<(usize, usize)> = Vec::new();
        let mut constrained_elements: HashSet<(usize, usize)> = HashSet::new();
        
        let mut components: Vec<Vec<(usize, usize)>> = Vec::new();
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(error) => panic!("Error while reading file at line {:?}", error),
            };
            if line == "" {
                if component.len() > 1 {
                    components.push(component);
                }
                component = Vec::new();
            } else {
                let mut split = line.split_whitespace();
                let seq_id = Constraints::parse_usize(split.next().unwrap());
                let timestamp = Constraints::parse_usize(split.next().unwrap());
                constrained_elements.insert((seq_id, timestamp));
                component.push((seq_id, timestamp));
            }
        }
        if component.len() > 1 {
            components.push(component);
        }
        Self { components: Array1::from_vec(components), constrained_elements}
    }

    pub fn get_component(&self, idx: usize) -> &Vec<(usize, usize)> {
        &self.components[idx]
    }
}
