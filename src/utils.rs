use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

type TTAG = Option<usize>;

pub fn load_sequences<const D: usize>(path: &PathBuf) -> Vec<Vec<[usize; D]>> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);

    let mut ret: Vec<Vec<[usize; D]>> = Vec::new();
    let mut last_id: Option<usize> = None;

    let mut cur_seq: Vec<[usize; D]> = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        let s: Vec<usize> = line.split(" ").map(|x| x.parse::<usize>().unwrap()).collect();
        let tid = Some(s[0]);
        if tid != last_id {
            if last_id.is_some() {
                ret.push(cur_seq);
                cur_seq = Vec::new();
            }
        }
        last_id = tid;
        let mut element: [usize; D] = [0; D];
        for i in 1..s.len() {
            element[i-1] = s[i];
        }
        cur_seq.push(element);
    }
    ret.push(cur_seq);
    ret
}

pub fn load_tags(path: &PathBuf) -> Vec<Vec<TTAG>> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);

    let mut ret: Vec<Vec<TTAG>> = Vec::new();
    let mut current: Vec<TTAG> = Vec::new();
    let mut last_id: Option<usize> = None;

    for line in reader.lines() {
        let line = line.unwrap();
        let s: Vec<&str> = line.split(" ").collect();
        let tid = Some(s[0].parse::<usize>().unwrap());
        if tid != last_id {
            if last_id.is_some() {
                ret.push(current);
                current = Vec::new();
            }
        }
        let v: Option<usize> = if s[1] == "-1" { None } else { Some(s[1].parse::<usize>().unwrap()) };
        current.push(v);
        last_id = tid;
    }
    ret.push(current);
    ret
}
