use ndarray::Array1;
use std::collections::{HashSet, HashMap};

use indextree::Arena;
use indextree::NodeId;

use super::hmm::HMM;
use super::utils::SuperSequence;

type Backtrack = (usize, usize, NodeId);

#[derive(Debug)]
struct DPEntry {
    pub time: usize,
    pub state: usize,
    pub cstr_paths: HashMap<NodeId, (f64, Option<Backtrack>)>,
}

impl DPEntry {

    pub fn new(time: usize, state: usize) -> Self {
        let cstr_paths: HashMap<NodeId, (f64, Option<Backtrack>)> = HashMap::new();
        Self {time, state, cstr_paths}
    }

    pub fn update(&mut self, cost: f64, from: Option<Backtrack>, cstr: NodeId) {
        let entry = self.cstr_paths.entry(cstr).or_insert((f64::NEG_INFINITY, None));
        if cost > entry.0 {
            *entry = (cost, from);
        }
    }

    pub fn len(&self) -> usize {
        self.cstr_paths.len()
    }
}

fn get_choice(arena: &mut Arena<(i32, i32)>, node_idx: NodeId, comp_choice: usize) -> Result<usize, String> {

    let mut iter = node_idx.ancestors(arena);
    let mut current = iter.next();

    while !current.is_none() {
        let (comp, choice) = &arena[current.unwrap()].get();
        let ucomp = *comp as usize;
        let uchoice = *choice as usize;
        if ucomp == comp_choice {
            return Ok(uchoice);
        }
        current = iter.next();
    }
    Err(format!("Did not find choice for constraint {} in tree", comp_choice))
}

pub fn dp_solving(hmm: &HMM, sequence: &SuperSequence) -> Array1<usize> {
    let mut table: HashMap<(i32, usize), DPEntry> = HashMap::with_capacity(sequence.len()*hmm.nstates()/10);

    let arena = &mut Arena::new();
    let root = arena.new_node((-1, -1));

    let mut dp_entry_source = DPEntry::new(0, 0);
    dp_entry_source.update(0.0, None, root);
    table.insert((-1, 0), dp_entry_source);

    let mut finished_constraints: Vec<bool> = (0..sequence.nb_cstr).map(|_| false).collect();
    let mut valid_states: Vec<HashSet<usize>> = (0..sequence.nb_cstr).map(|_| (0..hmm.nstates()).collect()).collect();


    for idx in 0..sequence.len() {
        let element = &sequence[idx];
        // Check if the layer is constrained?
        let is_constrained = element.constraint_component != -1;

        for state_to in 0..hmm.nstates() {
            if is_constrained && !valid_states[element.constraint_component as usize].contains(&state_to) {
                // In a previous layer of the same component, this state was not possible. So even
                // if the transition to this state is now possible, it will never be selected so we
                // can safely pass it
                continue;
            }
            if !element.can_be_emited(hmm, state_to) {
                if is_constrained {
                    valid_states[element.constraint_component as usize].remove(&state_to);
                }
                continue;
            }
            let mut dp_entry = DPEntry::new(idx, state_to);
            for state_from in 0..hmm.nstates() {
                let arc_cost = element.arc_p(hmm, state_from, state_to);
                if arc_cost == f64::NEG_INFINITY { continue; }
                match table.get_mut(&(idx as i32 - 1, state_from)) {
                    None => (),
                    Some(entry) => {
                        if !is_constrained {
                            for (cstr_node_id, value) in &entry.cstr_paths {
                                let cost = value.0 + arc_cost;
                                dp_entry.update(cost, Some((entry.time, entry.state, *cstr_node_id)), *cstr_node_id);
                            }
                        } else {
                            let ucomp = element.constraint_component as usize;
                            for (cstr_node_id, value) in &entry.cstr_paths {
                                let mut leaf = *cstr_node_id;
                                if idx == sequence.first_pos_cstr[ucomp] {
                                    let newnode = arena.new_node((element.constraint_component, state_to as i32));
                                    cstr_node_id.append(newnode, arena);
                                    leaf = newnode;
                                }
                                let choice = get_choice(arena, leaf, ucomp).unwrap();
                                if choice == state_to {
                                    let cost = value.0 + arc_cost;
                                    dp_entry.update(cost, Some((entry.time, entry.state, *cstr_node_id)), leaf);
                                }
                            }
                        }
                    }
                };
            }
            // Let assume that we saw the last elements of Ci, every path of constraints
            // looks like
            // <_, _, ..., i1, ...>
            // <_, _, ..., i2, ...>
            // <_, _, ..., i3, ...>
            // ...
            // <_, _, ..., in, ...>
            //
            // if for two different i's, the rest of the choices are the same, we can
            // discard the choices with the lowest value (since whatever happens in the
            // rest of the DAG, it will apply to both choices paths)

            let mut should_prune = false;
            if is_constrained && element.last_of_constraint {
                should_prune = true;
                finished_constraints[element.constraint_component as usize] = true;
            }

            if should_prune {
                let mut to_remove: Vec<NodeId> = Vec::new();
                let cstr_node_ids: Vec<&NodeId> = dp_entry.cstr_paths.keys().collect();
                for i in 0..cstr_node_ids.len() {
                    for j in i+1..cstr_node_ids.len() {
                        let n1 = cstr_node_ids[i];
                        let n2 = cstr_node_ids[j];
                        let mut same = true;
                        for k in 0..sequence.nb_cstr {
                            if sequence.first_pos_cstr[k] <= idx {
                                let c1 = get_choice(arena, *n1, k).unwrap();
                                let c2 = get_choice(arena, *n2, k).unwrap();
                                if !finished_constraints[k] && c1 != c2 {
                                    same = false;
                                    break;
                                }
                            }
                        }

                        // Only keep the best one
                        if same {
                            let v1 = dp_entry.cstr_paths.get(n1).unwrap().0;
                            let v2 = dp_entry.cstr_paths.get(n2).unwrap().0;
                            if v1 > v2 {
                                to_remove.push(n2.to_owned());
                            } else {
                                to_remove.push(n1.to_owned());
                            }
                        }
                    }
                }
                for tr in to_remove {
                    dp_entry.cstr_paths.remove(&tr);
                }
            }

            if dp_entry.len() > 0 {
                table.insert((idx as i32, state_to), dp_entry);
            }
        }
    }

    let mut dp_entry_target: Option<&DPEntry> = None;
    let mut best_cstr_path: Option<NodeId> = None;
    let mut best_cost = f64::NEG_INFINITY;
    for state in 0..hmm.nstates() {
        match table.get(&(sequence.len() as i32 - 1, state)) {
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
    
    let mut sol = Array1::from_elem(sequence.len(), 0);
    let mut current  = dp_entry_target.unwrap();
    for idx in (0..sol.len()).rev() {
        sol[idx] = current.state;
        let (_, from) = current.cstr_paths.get(&best_cstr_path.unwrap()).unwrap();
        if idx != 0 {
            let from = from.unwrap();
            let key = (from.0 as i32, from.1);
            best_cstr_path = Some(from.2);
            current = table.get(&key).unwrap();
        }
    }
    sol
}
