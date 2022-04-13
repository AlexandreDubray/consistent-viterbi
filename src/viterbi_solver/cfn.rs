use super::utils::SuperSequence;
use super::super::hmm::hmm::HMM;
use ndarray::prelude::*;
use ndarray::{Data, DataMut};
use ndarray_stats::QuantileExt;
use std::io::Write;
use std::fs::File;
use std::path::PathBuf;

fn longest_path<'a, const D: usize>(hmm: &'a HMM<D>, sequence: &'a SuperSequence<'a, D>, array: &mut Array2<f64>, t_from: usize, n_from: usize, t_to: usize, n_to: usize) -> f64 {
    array.row_mut(0).fill(f64::NEG_INFINITY);
    array[[0, n_from]] = 0.0;
    let mut t = 1;
    while t_from + t <= t_to {
        if sequence[t_from + t].is_constrained() {
            array.row_mut(t).fill(f64::NEG_INFINITY);
            let n = if t_from + t < t_to { n_from } else { n_to };
            let transitions = sequence[t_from + t].transitions(hmm, n);
            let trans_probs = &array.row(t - 1) + &transitions;
            let p = trans_probs.max().unwrap() + hmm.emit_prob(n, sequence[t_from + t].value);
            array[[t, n]] = p;
        } else {
            for n in 0..hmm.nstates() {
                let transitions = sequence[t_from + t].transitions(hmm, n);
                let trans_probs = &array.row(t - 1) + &transitions;
                let p = trans_probs.max().unwrap() + hmm.emit_prob(n, sequence[t_from + t].value);
                array[[t, n]] = p;
            }
        }
        t += 1;
    }
    *array.row(t-1).max().unwrap()
}

fn get_unary_start_cost<'a, const D: usize>(hmm: &'a HMM<D>, sequence: &'a SuperSequence<'a, D>, array: &mut Array2<f64>, t_limit: usize) -> Array1<f64> {
    let init_prob = hmm.init_probs(sequence[0].value);
    if t_limit == 0 {
        return init_prob;
    }
    array.row_mut(0).assign(&init_prob);
    for t in 1..t_limit + 1 {
        for n in 0..hmm.nstates() {
            let transitions = sequence[t].transitions(hmm, n);
            let trans_probs = &array.row(t - 1) + &transitions;
            let p = trans_probs.max().unwrap() + hmm.emit_prob(n, sequence[t].value);
            array[[t, n]] = p;
        }
    }
    Array1::from_shape_fn(hmm.nstates(), |n| -> f64 {
        array[[t_limit, n]]
    })
}

fn get_unary_end_cost<'a, const D: usize>(hmm: &'a HMM<D>, sequence: &'a SuperSequence<'a, D>, array: &mut Array2<f64>, t_start: usize) -> Array1<f64> {
    if t_start == sequence.len() - 1 {
        return Array::from_elem(hmm.nstates(), 0.0);
    }
    Array::from_shape_fn(hmm.nstates(), |n| -> f64 {
        array.row_mut(0).fill(f64::NEG_INFINITY);
        array[[0, n]] = 0.0;
        for t in t_start+1..sequence.len() {
            if sequence[t].is_constrained() {
                array.row_mut(t-t_start).fill(f64::NEG_INFINITY);
                let transitions = sequence[t].transitions(hmm, n);
                let trans_probs = &array.row(t-t_start-1) + &transitions;
                let p = trans_probs.max().unwrap() + hmm.emit_prob(n, sequence[t].value);
                array[[t-t_start, n]] = p;
            } else {
                for nn in 0..hmm.nstates() {
                    let transitions = sequence[t].transitions(hmm, nn);
                    let trans_probs = &array.row(t-t_start-1) + &transitions;
                    let p = trans_probs.max().unwrap() + hmm.emit_prob(nn, sequence[t].value);
                    array[[t-t_start, nn]] = p;
                }
            }
        }
        *array.row(sequence.len() - 1 - t_start).max().unwrap()
    })
}

pub fn write_cfn<'a, const D: usize>(hmm: &'a HMM<D>, sequence: &'a SuperSequence<'a, D>, outpath: &mut PathBuf) {
    let k = sequence.number_constraints();
    let mut constraint_boundaries: Vec<(usize, usize)> = Vec::new();

    let mut last_cid: Option<usize> = None;
    let mut longest_segment_size = 0;
    for t in 0..sequence.len() {
        if sequence[t].is_constrained() {
            let cid = sequence[t].constraint_component as usize;
            match last_cid {
                None => {
                    longest_segment_size = t + 1;
                    constraint_boundaries.push((t, cid));
                },
                Some(lcid) => {
                    if lcid != cid {
                        let segment_size = t - constraint_boundaries.last().unwrap().0 + 1;
                        if segment_size > longest_segment_size {
                            longest_segment_size = segment_size;
                        }
                        constraint_boundaries.push((t, cid));
                    }
                }
            };
            last_cid = Some(cid);
        }
    }
    let last_segment_size = sequence.len() - constraint_boundaries.last().unwrap().0 + 1;
    if last_segment_size > longest_segment_size {
        longest_segment_size = last_segment_size;
    }

    let n = hmm.nstates();
    let mut array = Array2::from_elem((longest_segment_size + 1, n), 0.0);
    let mut cost_tables = Array2::from_elem((k, k), Array2::from_elem((n, n), 0.0));

    for i in 0..constraint_boundaries.len() - 1 {
        let (t_from, cid_from) = &constraint_boundaries[i];
        let (t_to, cid_to) = &constraint_boundaries[i+1];
        for n1 in 0..n {
            for n2 in 0..n {
                let cost = longest_path(hmm, sequence, &mut array, *t_from, n1, *t_to, n2);
                if cost != f64::NEG_INFINITY {
                    if cost_tables[[*cid_from, *cid_to]][[n1, n2]] == 0.0 {
                        cost_tables[[*cid_from, *cid_to]][[n1, n2]] = cost;
                    } else {
                        cost_tables[[*cid_from, *cid_to]][[n1, n2]] += cost;
                    }
                    if cost_tables[[*cid_to, *cid_from]][[n2, n1]] == 0.0 {
                        cost_tables[[*cid_to, *cid_from]][[n2, n1]] = cost;
                    } else {
                        cost_tables[[*cid_to, *cid_from]][[n2, n1]] += cost;
                    }
                }
            }
        }
    }

    let mut unary_costs = Array1::from_elem(k, Array1::from_elem(n, 0.0));
    let (f_time, f_cid) = constraint_boundaries[0];
    let (l_time, l_cid) = *constraint_boundaries.last().unwrap();
    let start_cost = get_unary_start_cost(hmm, sequence, &mut array, f_time);
    unary_costs[f_cid] += &start_cost;
    let end_cost = get_unary_end_cost(hmm, sequence, &mut array, l_time);
    unary_costs[l_cid] += &end_cost;

    let mut lower_bound = -1.0;

    for k1 in 0..k {
        for k2 in k1+1..k {
            let m = *cost_tables[[k1, k2]].min().unwrap();
            lower_bound += m;
        }
    }


    outpath.set_file_name("cfn_problem");
    let mut file = File::create(outpath).unwrap();

    file.write_all(format!("{{\n\tproblem: {{ name: consistent_viterbi, mustbe: >{}}},\n", lower_bound).as_bytes()).unwrap();
    file.write_all("\tvariables: {".as_bytes()).unwrap();
    
    let mut s = String::from("[");
    for i in 0..hmm.nstates() {
        s.push_str(&format!("s{}", i));
        if i == hmm.nstates() - 1 {
            s.push_str("]");
        } else {
            s.push_str(",");
        }
    }


    for i in 0..k {
        file.write_all(format!("n{}: {}{}", i, s, if i != k-1 { "," } else {"},\n"}).as_bytes()).unwrap();
    }
    file.write_all("\tfunctions: {\n".as_bytes()).unwrap();
    for i in 0..k {
        file.write_all(format!("\t\tf{}: {{ scope: [n{}], costs: ", i, i).as_bytes()).unwrap();
        unary_costs[i].to_file(&mut file);
        file.write_all("}, \n".as_bytes()).unwrap();
        for j in i+1..k {
            let tcost = &cost_tables[[i, j]];
            if tcost.sum() != 0.0 {
                file.write_all(format!("\t\tf{}_{}: {{ scope: [n{}, n{}], costs: ", i, j, i, j).as_bytes()).unwrap();
                tcost.to_file(&mut file);
                file.write_all("},\n".as_bytes()).unwrap();
            }
        }
    }
    file.write_all("\t}\n}".as_bytes()).unwrap();
}

trait ArrayStr {
    fn to_file(&self, _: &mut File);
}

impl<S> ArrayStr for ArrayBase<S, Ix1>
where
    S: DataMut + Data<Elem = f64>,
{
    fn to_file(&self, file: &mut File) {
        file.write_all("[".as_bytes()).unwrap();
        for i in 0..self.len() {
            file.write(format!("{}{} ", self.get(i).unwrap(), if i == self.len() - 1 { "]" } else { "," }).as_bytes()).unwrap();
        }
    }
}

impl<S> ArrayStr for ArrayBase<S, Ix2>
where
    S: DataMut + Data<Elem = f64>,
{
    fn to_file(&self, file: &mut File) {
        file.write_all("[".as_bytes()).unwrap();
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                file.write(format!("{}{} ", self.row(i).get(j).unwrap(), if i == self.nrows() - 1 && j == self.ncols() - 1 { "]" } else { "," }).as_bytes()).unwrap();
            }
        }
    }
}
