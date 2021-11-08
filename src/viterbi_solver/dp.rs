use ndarray::Array1;
use std::collections::{HashSet, HashMap};
use std::rc::Rc;

use super::hmm::HMM;
use super::utils::SuperSequence;

type NodePtr = Rc<CstrNode>;
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

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct CstrNode {
    pub comp_id: usize,
    pub choice: usize,
    pub from: Option<Rc<CstrNode>>
}

impl CstrNode {

    pub fn new(comp_id: usize, choice: usize, from: Option<Rc<CstrNode>>) -> Self {
        Self { comp_id, choice, from }
    }

    pub fn get_choice(&self, comp_id: usize) -> Result<usize, String> {
        if self.comp_id == comp_id {
            Ok(self.choice)
        } else {
            let mut current = &self.from;
            while current.is_some() {
                let n = current.as_ref().unwrap();
                if n.comp_id == comp_id {
                    return Ok(n.choice)
                }
                current = &n.from;
            }
            Err(format!("Did not find choice for comp {}", comp_id))
        }
    }
}

pub struct DPSolver<'a> {
    pub solution: Array1<usize>,
    hmm: &'a HMM,
    sequence: &'a SuperSequence,
    table: HashMap<(i32, usize), DPEntry>,
}

impl<'b> DPSolver<'b> {

    pub fn new(hmm: &'b HMM, sequence: &'b SuperSequence) -> Self {
        let solution = Array1::from_elem(sequence.len(), 0);
        let table: HashMap<(i32, usize), DPEntry> = HashMap::new();
        Self {solution, hmm, sequence, table}
    }

    fn clear_table(&mut self, idx: usize, state: usize) {
        let mut current = Some(idx);
        while current.is_some() {
            self.table.remove(&(current.unwrap() as i32, state));
            current = self.sequence[current.unwrap()].previous_idx_cstr;
        }
    }

    pub fn dp_solving(&mut self) {
        let root = Rc::new(CstrNode::new(self.sequence.nb_cstr, 0, None));

        let mut dp_entry_source = DPEntry::new(0, 0);
        dp_entry_source.update(0.0, None, root);
        self.table.insert((-1, 0), dp_entry_source);

        let mut finished_constraints: Vec<bool> = (0..self.sequence.nb_cstr).map(|_| false).collect();
        let mut valid_states: Vec<HashSet<usize>> = (0..self.sequence.nb_cstr).map(|_| (0..self.hmm.nstates()).collect()).collect();

        for idx in 0.. self.sequence.len() {
            if idx % 1000 == 0 {
                println!("{}/{}", idx, self.sequence.len());
            }
            let element = &self.sequence[idx];
            // Check if the layer is constrained?
            let is_constrained = element.constraint_component != -1;

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
                                        Rc::new(CstrNode::new(ucomp, state_to, Some(Rc::clone(cstr_node_id))))
                                    } else {
                                        Rc::clone(cstr_node_id)
                                    };
                                    let choice = leaf.get_choice(ucomp).unwrap();
                                    if choice == state_to {
                                        let cost = value.0 + arc_cost;
                                        dp_entry.update(cost, Some((entry.time, entry.state, Rc::clone(cstr_node_id))), Rc::clone(&leaf));
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
                    let mut to_remove: Vec<Rc<CstrNode>> = Vec::new();
                    let cstr_node_ids: Vec<&NodePtr> = dp_entry.cstr_paths.keys().collect();
                    for i in 0..cstr_node_ids.len() {
                        for j in i+1..cstr_node_ids.len() {
                            let n1 = cstr_node_ids[i];
                            let n2 = cstr_node_ids[j];
                            let mut same = true;
                            for k in 0..self.sequence.nb_cstr {
                                if self.sequence.first_pos_cstr[k] <= idx {
                                    let c1 = n1.get_choice(k).unwrap();
                                    let c2 = n2.get_choice(k).unwrap();
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
                                    to_remove.push(Rc::clone(n2));
                                } else {
                                    to_remove.push(Rc::clone(n1));
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
