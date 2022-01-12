use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::fs::File;
use std::collections::HashSet;

pub struct Constraints {
    pub components: Vec<HashSet<(usize, usize)>>,
}

impl Constraints {

    pub fn from_file(path: &PathBuf) -> Self {
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);

        // Each component is separated by an empty line
        let mut component: HashSet<(usize, usize)> = HashSet::new();
        
        let mut components: Vec<HashSet<(usize, usize)>> = Vec::new();
        for line in reader.lines() {
            let line = line.unwrap();
            if line == "" {
                if component.len() > 1 {
                    components.push(component);
                }
                component = HashSet::new();
            } else {
                let mut split = line.split_whitespace();
                let seq_id = split.next().unwrap().parse::<usize>().unwrap();
                let timestamp = split.next().unwrap().parse::<usize>().unwrap();
                component.insert((seq_id, timestamp));
            }
        }
        if component.len() > 1 {
            components.push(component);
        }
        Self {components}
    }

    pub fn from_tags(truth: &Vec<Vec<Option<usize>>>) -> Self {
        let mut comp_value: Vec<usize> = Vec::new();
        let mut components: Vec<HashSet<(usize, usize)>> = Vec::new();
        for seq_id in 0..truth.len() {
            let tags = &truth[seq_id];
            for t in 0..tags.len() {
                if let Some(tag) = tags[t] {
                    let mut comp_id: Option<usize> = None;
                    for i in 0..comp_value.len() {
                        if comp_value[i] == tag {
                            comp_id = Some(i);
                            break;
                        }
                    }
                    match comp_id {
                        Some(id) => {
                            components[id].insert((seq_id, t));
                        }
                        None => {
                            comp_value.push(tag);
                            components.push(HashSet::new());
                            let id = components.len()-1;
                            components[id].insert((seq_id, t));
                        }
                    };
                }
            }
        }
        Self {components}
    }
}
