use ndarray::Array1;
use std::collections::HashMap;

use indextree::Arena;
use indextree::NodeId;

use super::hmm::HMM;
use super::constraints::Constraints;

struct DPEntry {
    pub time: usize,
    pub state: usize,
    pub cstr_paths: HashMap<NodeId, (f64, Option<(usize, usize, NodeId)>)>,
}

impl DPEntry {

    pub fn new(time: usize, state: usize) -> Self {
        let cstr_paths: HashMap<NodeId, (f64, Option<(usize, usize, NodeId)>)> = HashMap::new();
        Self {time, state, cstr_paths}
    }

    pub fn update(&mut self, cost: f64, from: Option<(usize, usize, NodeId)>, cstr: NodeId) {
        let entry = self.cstr_paths.entry(cstr).or_insert((f64::NEG_INFINITY, None));
        if cost > entry.0 {
            *entry = (cost, from);
        }
    }
}

fn get_sequences_ordering(sequences: &Array1<Array1<usize>>, constraints: &Constraints) -> Vec<(usize, usize)> {
    let mut nb_elem_cstr: Vec<(usize, usize)> = (0..sequences.len()).map(|seq_id| -> (usize, usize) {
        let mut count = 0;
        for t in 0..sequences[seq_id].len() {
            if constraints.get_comp_id(seq_id, t) != -1 {
                count += 1;
            }
        }
        (count, seq_id)
    }).collect();
    nb_elem_cstr.sort();
    nb_elem_cstr
}

fn get_child(arena: &mut Arena<Array1<i32>>, node_idx: NodeId, comp_choice: usize, state_choice: usize) -> NodeId {
    let node = &arena[node_idx];

    if node.get()[comp_choice] != -1 {
        return node_idx;
    }

    let has_children = node.first_child().is_none();

    if has_children {
        let mut current_child = node.first_child();
        while !current_child.is_none() {
            let current_id = current_child.unwrap();
            let child = &arena[current_id];
            if child.get()[comp_choice] == state_choice as i32 {
                return current_id;
            }
            current_child = child.next_sibling();
        }
    }
    let mut new_choices = node.get().clone();
    assert!(new_choices[comp_choice] == -1);
    new_choices[comp_choice] = state_choice as i32;
    let new_id = arena.new_node(new_choices);
    node_idx.append(new_id, arena);
    new_id
}

pub fn dp_solving(hmm: &HMM, sequences: &Array1<Array1<usize>>, constraints: &Constraints) -> Array1<Array1<usize>> {
    let total_length = sequences.map(|x| -> usize { x.len() }).sum();
    let mut table: HashMap<(usize, usize), DPEntry> = HashMap::with_capacity(total_length*hmm.nstates());
    // initialize the DPEntry for the first layer of the first sequence
    let arena = &mut Arena::new();
    let root = arena.new_node(Array1::from_elem(hmm.nstates(), -1));

    let mut dp_entry_source = DPEntry::new(0, 0);
    dp_entry_source.update(0.0, None, root);
    table.insert((0, 0), dp_entry_source);

    let ordering = get_sequences_ordering(sequences, constraints);


    let mut idx = 0;
    for (_, seq_id) in &ordering {
        let sequence = &sequences[*seq_id];
        for t in 0..sequence.len() {
            if idx % 100 == 0 {
                println!("{}/{} {} nodes in the cstr_tree", idx, total_length, arena.count());
            }
            idx += 1;
            // Check if the layer is constrained?
            let comp_id = constraints.get_comp_id(*seq_id, t);
            let is_constrained = comp_id != -1;

            for state_to in 0..hmm.nstates() {
                let emit_prob = hmm.emit_prob(state_to, sequence[t]);
                if emit_prob > f64::NEG_INFINITY {
                    let mut dp_entry = DPEntry::new(idx, state_to);
                    let mut updated = false;
                    for state_from in 0..hmm.nstates() {
                        let arc_cost = if t == 0 { hmm.init_prob(state_to, sequence[t]) } else { hmm.transition_prob(state_from, state_to, sequence[t]) };
                        if arc_cost > f64::NEG_INFINITY {
                            match table.get_mut(&(idx - 1, state_from)) {
                                None => (),
                                Some(entry) => {
                                    if !is_constrained {
                                        for (cstr_node_id, value) in &entry.cstr_paths {
                                            let cost = value.0 + arc_cost;
                                            dp_entry.update(cost, Some((entry.time, entry.state, *cstr_node_id)), *cstr_node_id);
                                            updated = true;
                                        }
                                    } else {
                                        for (cstr_node_id, value) in &entry.cstr_paths {
                                            let node_id = get_child(arena, *cstr_node_id, comp_id as usize, state_to);
                                            let node = &arena[node_id];
                                            if node.get()[comp_id as usize] == state_to as i32 {
                                                let cost = value.0 + arc_cost;
                                                dp_entry.update(cost, Some((entry.time, entry.state, *cstr_node_id)), node_id);
                                                updated = true;
                                            }
                                        }
                                    }
                                }
                            };
                        }
                    }
                    if updated {
                        table.insert((idx, state_to), dp_entry);
                    }
                }
            }
        }
    }

    let mut dp_entry_target: Option<&DPEntry> = None;
    let mut best_cstr_path: Option<NodeId> = None;
    let mut best_cost = f64::NEG_INFINITY;
    for state in 0..hmm.nstates() {
        match table.get(&(idx, state)) {
            Some(entry) => {
                for (cstr_path_idx, value) in &entry.cstr_paths {
                    if value.0 > best_cost {
                        best_cost = value.0;
                        dp_entry_target = Some(entry);
                        best_cstr_path = Some(*cstr_path_idx);
                    }
                }
            },
            None => ()
        };
    }
    
    let mut sol = sequences.map(|x| -> Array1<usize> { Array1::from_elem(x.len(), 0) });
    let mut current  = dp_entry_target.unwrap();
    for i in (0..ordering.len()).rev() {
        let seq_id = ordering[i].1;
        for t in (0..sequences[seq_id].len()).rev() {
            sol[seq_id][t] = current.state;
            let (_, from) = current.cstr_paths.get(&best_cstr_path.unwrap()).unwrap();
            let from = from.unwrap();
            let key = (from.0, from.1);
            best_cstr_path = Some(from.2);
            current = table.get(&key).unwrap();
        }
    }
    sol
}
