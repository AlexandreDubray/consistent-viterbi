use ndarray::Array1;
use std::collections::HashMap;
use rand::Rng;

use super::hmm::HMM;
use super::constraints::Constraints;

struct DPEntry {
    pub time: usize,
    pub state: usize,
    pub cstr_paths: HashMap<usize, (f64, Option<(usize, usize, usize)>)>,
}

impl DPEntry {

    pub fn new(time: usize, state: usize) -> Self {
        let cstr_paths: HashMap<usize, (f64, Option<(usize, usize, usize)>)> = HashMap::new();
        Self {time, state, cstr_paths}
    }

    pub fn update(&mut self, cost: f64, from: Option<(usize, usize, usize)>, cstr_path_idx: usize) {
        let entry = self.cstr_paths.entry(cstr_path_idx).or_insert((f64::NEG_INFINITY, None));
        if cost > entry.0 {
            *entry = (cost, from);
        }
    }
}

pub fn dp_solving(hmm: &HMM, sequences: &Array1<Array1<usize>>, constraints: &Constraints, prop_cstr: f64) -> Array1<Array1<usize>> {
    let total_length = sequences.map(|x| -> usize { x.len() }).sum();
    let mut table: HashMap<(usize, usize), DPEntry> = HashMap::with_capacity(total_length*hmm.nstates());
    // initialize the DPEntry for the first layer of the first sequence
    let mut cstr_paths: Vec<Array1<i32>> = Vec::new();
    cstr_paths.push(Array1::from_elem(constraints.components.len(), -1));
    let mut dp_entry_source = DPEntry::new(0, 0);
    dp_entry_source.update(0.0, None, 0);
    table.insert((0, 0), dp_entry_source);

    let mut rng = rand::thread_rng();
    
    let mut idx = 0;
    for seq_id in 0..sequences.len() {
        let sequence = &sequences[seq_id];
        for t in 0..sequence.len() {
            if idx % 10000 == 0 {
                println!("{}/{}", idx, total_length);
            }
            idx += 1;
            // Check if the layer is constrained?
            let comp_id = constraints.get_comp_id(seq_id, t);
            let prop_t: f64 = rng.gen();
            let is_constrained = prop_t < prop_cstr && comp_id != -1;

            if is_constrained {
                for state_to in 0..hmm.nstates() {
                    if hmm.emit_prob(state_to, sequence[t]) > f64::NEG_INFINITY {
                        let mut dp_entry = DPEntry::new(idx, state_to);
                        let mut updated = false;
                        for state_from in 0..hmm.nstates() {
                            match table.get_mut(&(idx-1, state_from)) {
                                Some(entry) => {
                                    for (cstr_path_idx, value) in &entry.cstr_paths {
                                        let choice = cstr_paths[*cstr_path_idx][comp_id as usize];
                                        let arc_cost = if t == 0 { hmm.init_prob(state_to, sequence[t]) } else { hmm.transition_prob(state_from, state_to, sequence[t]) };
                                        let cost = value.0 + arc_cost;
                                        if choice != -1 && choice == state_to as i32 {
                                            dp_entry.update(cost, Some((entry.time, entry.state, *cstr_path_idx)), *cstr_path_idx);
                                            updated = true;
                                        } else if choice == -1 {
                                            // New cstr_node with choice = state_to
                                            let mut new_cstr_array = cstr_paths[*cstr_path_idx].clone();
                                            new_cstr_array[comp_id as usize] = state_to as i32;
                                            cstr_paths.push(new_cstr_array);
                                            dp_entry.update(cost, Some((entry.time, entry.state, *cstr_path_idx)), cstr_paths.len()-1);
                                            updated = true;
                                        }
                                    }
                                },
                                None => ()
                            };
                        }
                        if updated {
                            table.insert((idx, state_to), dp_entry);
                        }
                    }
                }
            } else {
                for state_to in 0..hmm.nstates() {
                    if hmm.emit_prob(state_to, sequence[t]) > f64::NEG_INFINITY {
                        let mut dp_entry = DPEntry::new(idx, state_to);
                        let mut updated = false;
                        for state_from in 0..hmm.nstates() {
                            match table.get(&(idx-1, state_from)) {
                                Some(entry) => {
                                    for (cstr_path_idx, value) in &entry.cstr_paths {
                                        let arc_cost = if t == 0 { hmm.init_prob(state_to, sequence[t]) } else { hmm.transition_prob(state_from, state_to, sequence[t]) };
                                        let cost = value.0 + arc_cost;
                                        dp_entry.update(cost, Some((entry.time, entry.state, *cstr_path_idx)), *cstr_path_idx);
                                        updated = true;
                                    }
                                },
                                None => ()
                            };
                        }
                        if updated {
                            table.insert((idx, state_to), dp_entry);
                        }
                    }
                }
            }
        }
    }

    let mut dp_entry_target: Option<&DPEntry> = None;
    let mut best_cstr_path = 0;
    let mut best_cost = f64::NEG_INFINITY;
    for state in 0..hmm.nstates() {
        match table.get(&(total_length, state)) {
            Some(entry) => {
                for (cstr_path_idx, value) in &entry.cstr_paths {
                    if value.0 > best_cost {
                        best_cost = value.0;
                        dp_entry_target = Some(entry);
                        best_cstr_path = *cstr_path_idx;
                    }
                }
            },
            None => ()
        };
    }
    
    let mut sol = sequences.map(|x| -> Array1<usize> { Array1::from_elem(x.len(), 0) });
    let mut current  = dp_entry_target.unwrap();
    for seq_id in (0..sequences.len()).rev() {
        for t in (0..sequences[seq_id].len()).rev() {
            sol[seq_id][t] = current.state;
            let (_, from) = current.cstr_paths.get(&best_cstr_path).unwrap();
            let from = from.unwrap();
            let key = (from.0, from.1);
            best_cstr_path = from.2;
            current = table.get(&key).unwrap();
        }
    }
    sol
}
