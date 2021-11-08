use ndarray::{Array1, Array2};
use std::collections::{HashSet, HashMap};
use std::rc::Rc;

use indextree::Arena;
use indextree::NodeId;

use super::hmm::HMM;
use super::utils::SuperSequence;

type NodePtr = Rc<NodeId>;
type Backtrack = (usize, usize, NodePtr);

#[derive(Debug)]
struct DPEntry {
    pub time: usize,
    pub state: usize,
    pub cstr_paths: HashMap<NodePtr, (f64, Option<Backtrack>)>,
}

impl DPEntry {

    pub fn new(time: usize, state: usize) -> Self {
        let cstr_paths: HashMap<NodePtr, (f64, Option<Backtrack>)> = HashMap::new();
        Self {time, state, cstr_paths}
    }

    pub fn update(&mut self, cost: f64, from: Option<Backtrack>, cstr: NodePtr) {
        let entry = self.cstr_paths.entry(cstr).or_insert((f64::NEG_INFINITY, None));
        if cost > entry.0 {
            *entry = (cost, from);
        }
    }

    pub fn len(&self) -> usize {
        self.cstr_paths.len()
    }
}

pub struct DPSolver<'a> {
    pub solution: Array1<usize>,
    hmm: &'a HMM,
    sequence: &'a SuperSequence,
    table: HashMap<(i32, usize), DPEntry>,
    node_arena: Arena<(i32, i32)>
}

impl<'b> DPSolver<'b> {

    pub fn new(hmm: &'b HMM, sequence: &'b SuperSequence) -> Self {
        let solution = Array1::from_elem(sequence.len(), 0);
        let table: HashMap<(i32, usize), DPEntry> = HashMap::new();
        let node_arena: Arena<(i32, i32)> = Arena::new();
        Self {solution, hmm, sequence, table, node_arena}
    }

    fn get_choice(&self, node_idx: &NodePtr, comp_choice: usize) -> Result<usize, String> {

        let mut iter = node_idx.ancestors(&self.node_arena);
        let mut current = iter.next();

        while !current.is_none() {
            let (comp, choice) = &self.node_arena[current.unwrap()].get();
            let ucomp = *comp as usize;
            let uchoice = *choice as usize;
            if ucomp == comp_choice {
                return Ok(uchoice);
            }
            current = iter.next();
        }
        Err(format!("Did not find choice for constraint {} in tree", comp_choice))
    }

    pub fn bender_decomposition(&mut self) {
        // Solve the initial problem without any constraints
        let mut active_constraints = Array1::from_elem(self.sequence.nb_cstr, false);
        let mut sequence_to_recompute = Array1::from_elem(self.sequence.nb_seqs, true);
        self.dp_solving(&active_constraints);
        sequence_to_recompute.fill(false);
        for i in 0..self.sequence.nb_cstr {

            let mut count = Array2::from_elem((self.sequence.nb_cstr, self.hmm.nstates()), 0);
            let mut max_val = Array1::from_elem(self.sequence.nb_cstr, 0);
            let mut sum = Array1::from_elem(self.sequence.nb_cstr, 0);

            for t in 0..self.sequence.len() {
                let comp_id = self.sequence[t].constraint_component;
                if comp_id != -1 {
                    let ucomp = comp_id as usize;
                    let tag = self.solution[t];
                    count[[ucomp, tag]] += 1;
                    if count[[ucomp, tag]] > max_val[ucomp] {
                        max_val[ucomp] = count[[ucomp, tag]];
                    }
                    sum[ucomp] += 1;
                }
            }
            let mut optimum = true;
            let mut next_added_cstr = 0;
            let mut max_viol = 0;
            let mut nb_violated = 0;
            for comp_id in 0..self.sequence.nb_cstr {
                if sum[comp_id] != max_val[comp_id] {
                    optimum = false;
                    nb_violated += 1;
                    let violations = sum[comp_id] - max_val[comp_id];
                    if !active_constraints[comp_id] && violations > max_viol {
                        next_added_cstr = comp_id;
                        max_viol = violations;
                    }
                }
            }
            if optimum {
                break;
            }
            println!("Adding constraints {} to active constraints ({} constraints violated)", next_added_cstr, nb_violated);
            active_constraints[next_added_cstr] = true;
            for seq_id in &self.sequence.seq_per_cstr[next_added_cstr] {
                sequence_to_recompute[*seq_id] = true;
            }
            println!("Run {}/{} max", i, self.sequence.nb_cstr);
            self.dp_solving(&active_constraints);
        }
    }

    fn clear_table(&mut self, idx: usize, state: usize) {
        let mut current = Some(idx);
        while current.is_some() {
            self.table.remove(&(current.unwrap() as i32, state));
            current = self.sequence[current.unwrap()].previous_idx_cstr;
        }
    }

    pub fn dp_solving(&mut self, active_constraints: &Array1<bool>) {
        let root = Rc::new(self.node_arena.new_node((-1, -1)));

        let mut dp_entry_source = DPEntry::new(0, 0);
        dp_entry_source.update(0.0, None, root);
        self.table.insert((-1, 0), dp_entry_source);

        let mut finished_constraints: Vec<bool> = (0..self.sequence.nb_cstr).map(|_| false).collect();
        let mut valid_states: Vec<HashSet<usize>> = (0..self.sequence.nb_cstr).map(|_| (0..self.hmm.nstates()).collect()).collect();

        for idx in 0.. self.sequence.len() {
            if idx % 10000 == 0 {
                println!("{}/{} (size of table is {})", idx, self.sequence.len(), self.table.len());
            }
            let element = &self.sequence[idx];
            // Check if the layer is constrained?
            let is_constrained = element.constraint_component != -1 && active_constraints[element.constraint_component as usize];

            for state_to in 0..self.hmm.nstates() {
                if is_constrained && !valid_states[element.constraint_component as usize].contains(&state_to) {
                    // In a previous layer of the same component, this state was not possible. So even
                    // if the transition to this state is now possible, it will never be selected so we
                    // can safely pass it
                    continue;
                }
                if !element.can_be_emited(self.hmm, state_to) {
                    if is_constrained {
                        valid_states[element.constraint_component as usize].remove(&state_to);
                        self.clear_table(idx, state_to);
                    }
                    continue;
                }
                let mut dp_entry = DPEntry::new(idx, state_to);
                for state_from in 0..self.hmm.nstates() {
                    let arc_cost = element.arc_p(self.hmm, state_from, state_to);
                    if arc_cost == f64::NEG_INFINITY { continue; }
                    match self.table.get_mut(&(idx as i32 - 1, state_from)) {
                        None => (),
                        Some(entry) => {
                            if !is_constrained {
                                for (cstr_node_id, value) in &entry.cstr_paths {
                                    let cost = value.0 + arc_cost;
                                    dp_entry.update(cost, Some((entry.time, entry.state, Rc::clone(cstr_node_id))), Rc::clone(cstr_node_id));
                                }
                            } else {
                                let ucomp = element.constraint_component as usize;
                                for (cstr_node_id, value) in &entry.cstr_paths {
                                    let leaf = if idx == self.sequence.first_pos_cstr[ucomp] {
                                        let newnode = self.node_arena.new_node((element.constraint_component, state_to as i32));
                                        cstr_node_id.append(newnode, &mut self.node_arena);
                                        Rc::new(newnode)
                                    } else {
                                        Rc::clone(cstr_node_id)
                                    };

                                    let mut iter = leaf.ancestors(&self.node_arena);
                                    let mut current = iter.next();
                                    while !current.is_none() {
                                        let (comp, choice) = self.node_arena[current.unwrap()].get();
                                        if ucomp == *comp as usize {
                                            if *choice as usize == state_to {
                                                let cost = value.0 + arc_cost;
                                                dp_entry.update(cost, Some((entry.time, entry.state, Rc::clone(cstr_node_id))), Rc::clone(&leaf));
                                            }
                                            break;
                                        }
                                        current = iter.next();
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
                    let cstr_node_ids: Vec<&NodePtr> = dp_entry.cstr_paths.keys().collect();
                    for i in 0..cstr_node_ids.len() {
                        for j in i+1..cstr_node_ids.len() {
                            let n1 = cstr_node_ids[i];
                            let n2 = cstr_node_ids[j];
                            let mut same = true;
                            for k in 0..self.sequence.nb_cstr {
                                if self.sequence.first_pos_cstr[k] <= idx {
                                    let c1 = self.get_choice(n1, k).unwrap();
                                    let c2 = self.get_choice(n2, k).unwrap();
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
                                    to_remove.push(*n2.to_owned());
                                } else {
                                    to_remove.push(*n1.to_owned());
                                }
                            }
                        }
                    }
                    for tr in to_remove {
                        dp_entry.cstr_paths.remove(&tr);
                    }
                }

                if dp_entry.len() > 0 {
                    self.table.insert((idx as i32, state_to), dp_entry);
                } else if is_constrained {
                    valid_states[element.constraint_component as usize].remove(&state_to);
                    self.clear_table(idx, state_to);
                }
            }
        }

        let mut dp_entry_target: Option<&DPEntry> = None;
        let mut best_cstr_path: Option<&NodePtr> = None;
        let mut best_cost = f64::NEG_INFINITY;
        for state in 0..self.hmm.nstates() {
            match self.table.get(&(self.sequence.len() as i32 - 1, state)) {
                Some(entry) => {
                    for (cstr_path_idx, value) in &entry.cstr_paths {
                        if value.0 > best_cost {
                            best_cost = value.0;
                            dp_entry_target = Some(entry);
                            best_cstr_path = Some(cstr_path_idx);
                        }
                    }
                },
                None => ()
            };
        }
        
        let mut current  = dp_entry_target.unwrap();
        for idx in (0..self.solution.len()).rev() {
            self.solution[idx] = current.state;
            let (_, from) = current.cstr_paths.get(best_cstr_path.unwrap()).unwrap();
            if idx != 0 {
                let from = from.as_ref().unwrap();
                let key = (from.0 as i32, from.1);
                best_cstr_path = Some(&from.2);
                current = self.table.get(&key).unwrap();
            }
        }
    }
}
