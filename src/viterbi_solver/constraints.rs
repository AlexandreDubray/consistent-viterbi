use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::fs::File;
use std::collections::HashMap;

use rand::prelude::*;


pub struct Constraints {
    pub components: Vec<Vec<(usize, usize)>>,
    pub last_elements: Vec<(usize, usize)>,
    full_components: Vec<Vec<(usize, usize)>>,
    map_elem_comp_id: HashMap<(usize, usize), usize>,
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
        
        let mut full_components: Vec<Vec<(usize, usize)>> = Vec::new();
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(error) => panic!("Error while reading file at line {:?}", error),
            };
            if line == "" {
                if component.len() > 1 {
                    full_components.push(component);
                }
                component = Vec::new();
            } else {
                let mut split = line.split_whitespace();
                let seq_id = Constraints::parse_usize(split.next().unwrap());
                let timestamp = Constraints::parse_usize(split.next().unwrap());
                component.push((seq_id, timestamp));
            }
        }
        if component.len() > 1 {
            full_components.push(component);
        }
        let components: Vec<Vec<(usize, usize)>> = Vec::new();
        let map_elem_comp_id: HashMap<(usize, usize), usize> = HashMap::new();
        let last_elements: Vec<(usize, usize)> = Vec::new();
        Self {components, last_elements, full_components, map_elem_comp_id}
    }

    pub fn keep_prop(&mut self, prop: f64) {
        self.components.clear();
        self.last_elements.clear();
        self.map_elem_comp_id.clear();

        let mut rng = thread_rng();
        let mut current_comp: Vec<(usize, usize)> = Vec::new();
        for i in 0..self.full_components.len() {
            for elem in &self.full_components[i] {
                if rng.gen::<f64>() < prop {
                    current_comp.push(*elem);
                    self.map_elem_comp_id.insert(*elem, self.components.len());
                }
            }
            if current_comp.len() > 1 {
                self.last_elements.push(*current_comp.last().unwrap());
                self.components.push(current_comp);
                current_comp = Vec::new();
            } else if current_comp.len() == 1 {
                self.map_elem_comp_id.remove(&current_comp[0]);
            }
        }
    }

    pub fn get_comp_id(&self, seq_id: usize, time: usize) -> i32 {
        match self.map_elem_comp_id.get(&(seq_id, time)) {
            Some(x) => *x as i32,
            None => -1
        }
    }
}
