use std::io::{BufRead, BufReader};
use std::fs::File;

struct ConsistencyConstraint {
    seq_id_from: usize,
    seq_id_to: usize,
    timestamp_from: usize,
    timestamp_to: usize
}

pub struct Constraints {
    pub components: Vec<Vec<(usize, usize)>>
}

impl Constraints {

    fn parse_usize(s: &str) -> usize {
        match s.parse::<usize>() {
            Ok(v) => v,
            Err(error) => panic!("Can not parse {} as usize: {:?}", s, error)
        }
    }

    pub fn from_file(filename: &str) -> Self {
        let file = match File::open(filename) {
            Ok(f) => f,
            Err(error) => panic!("Unable to open constraints file {}: {:?}", filename, error)
        };
        let reader = BufReader::new(file);
        // Each component is separated by an empty line
        let mut component: Vec<(usize, usize)> = Vec::new();
        
        let mut components: Vec<Vec<(usize, usize)>> = Vec::new();
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(error) => panic!("Error while reading file at line {:?}", error),
            };
            if line == "" {
                components.push(component);
                component = Vec::new();
            } else {
                let mut split = line.split_whitespace();
                let seq_id = Constraints::parse_usize(split.next().unwrap());
                let timestamp = Constraints::parse_usize(split.next().unwrap());
                component.push((seq_id, timestamp));
            }
        }
        Self { components }
    }

    pub fn get_component(&self, idx: usize) -> &Vec<(usize, usize)> {
        &self.components[idx]
    }
}
