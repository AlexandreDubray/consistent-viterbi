use ndarray::Array1;
use std::collections::{HashSet, HashMap};

use super::super::hmm::hmm::HMM;
use super::utils::SuperSequence;

fn get_cstr_choice(uid: usize, cstr_id: usize, c: usize) -> usize {
    // bad
    // (uid / c.pow(cstr_id as u32)) % c
    // Good
    (cstr_id / c.pow(uid as u32)) % c
}

fn update_cstr_choice(uid: usize, cstr_id: usize, choice: usize, c: usize) -> usize {
    let current_choice = get_cstr_choice(uid, cstr_id, c);
    assert!(current_choice == 0);
    // bad
    //uid + c.pow(choice as u32)
    cstr_id + choice*c.pow(uid as u32)
}


pub struct DPSolver<'a, const D: usize> {
    pub solution: Array1<usize>,
    hmm: &'a HMM<D>,
    sequence: &'a mut SuperSequence<'a, D>,
    table: Array1<HashMap<(usize, usize), (usize, usize, f64)>>,
    pub objective: f64
}


impl<'b, const D: usize> DPSolver<'b, D> {

    pub fn new(hmm: &'b HMM<D>, sequence: &'b mut SuperSequence<'b, D>) -> Self {
        let solution = Array1::from_elem(sequence.len(), 0);
        let table: Array1<HashMap<(usize, usize), (usize, usize, f64)>> = Array1::from_elem(sequence.len(), HashMap::new());
        Self {solution, hmm, sequence, table, objective: 0.0}
    }

    fn prune(&mut self, t: usize, finished_constraints: &Vec<bool>) {
        let mut to_remove: Vec<(usize, usize)> = Vec::new();
        let keys: Vec<&(usize, usize)> = self.table[t].keys().collect();
        let c = self.hmm.nstates() + 1;
        for i in 0..keys.len() {
            for j in i+1..keys.len() {
                let (s1, cstr1) = keys[i];
                let (s2, cstr2) = keys[j];
                if s1 == s2 {
                    let mut same = true;
                    for k in 0..self.sequence.nb_cstr {
                        if !finished_constraints[k] {
                            same = get_cstr_choice(*cstr1, k, c) == get_cstr_choice(*cstr2, k, c);
                            if !same { break; }
                        }
                    }
                    
                    if same {
                        let v1 = self.table[t].get(&keys[i]).unwrap().2;
                        let v2 = self.table[t].get(&keys[j]).unwrap().2;
                        if v1 > v2 {
                            to_remove.push((*s2, *cstr2));
                        } else {
                            to_remove.push((*s1, *cstr1));
                        }
                    }
                }
            }
        }

        for tr in to_remove {
            self.table[t].remove(&tr);
        }
    }

    pub fn dp_solving(&mut self) {
        // There are nstates but the fact that a component has no assigned state is represented by
        // 0, thus we need nstates + 1 integers
        let nb_cstr_choice = self.hmm.nstates() + 1;

        let mut finished_constraints: Vec<bool> = (0..self.sequence.nb_cstr).map(|_| false).collect();
        let mut valid_states: Vec<HashSet<usize>> = (0..self.sequence.nb_cstr).map(|_| (0..self.hmm.nstates()).collect()).collect();

        {
            let mut to_insert: Vec<((usize, usize), (usize, usize, f64))> = Vec::new();
            let first = &self.sequence[0];
            for state in 0..self.hmm.nstates() {
                let is_constrained = self.sequence.is_constrained(0);
                if !first.can_be_emited(self.hmm, state) { continue; }
                let cost = first.arc_p(self.hmm, 0, state);
                if cost > f64::NEG_INFINITY {
                    let cstr = if is_constrained {
                        let ucomp = first.constraint_component as usize;
                        update_cstr_choice(0, ucomp, state, nb_cstr_choice)
                    } else {
                        0
                    };
                    to_insert.push(((state, cstr), (0, 0, cost)));
                }
            }

            for v in to_insert {
                let key = v.0;
                let value = v.1;
                self.table[0].insert(key, value);
            }
        }

        for idx in 1.. self.sequence.len() {
            let element = &self.sequence[idx];
            let is_constrained = self.sequence.is_constrained(idx);

            let mut updated = false;
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
                    }
                    continue;
                }

                let mut to_insert: Vec<((usize, usize), (usize, usize, f64))> = Vec::new();
                for (key, val) in self.table[idx - 1].iter() {
                    let state_from = key.0;
                    let arc_cost = element.arc_p(self.hmm, state_from, state_to);
                    if arc_cost == f64::NEG_INFINITY { continue; }
                    let cost = arc_cost + val.2;
                    let cstr = key.1;
                    if !is_constrained {
                        to_insert.push(((state_to, cstr), (state_from, cstr, cost)));
                    } else {
                        let ucomp = element.constraint_component as usize;
                        let choice = get_cstr_choice(ucomp, cstr, nb_cstr_choice);
                        if choice == 0 {
                            // No choice made for the constraint component so far
                            let new_cstr = update_cstr_choice(ucomp, cstr, state_to + 1, nb_cstr_choice);
                            to_insert.push(((state_to, new_cstr), (state_from, cstr, cost)));
                        } else if choice == state_to + 1 {
                            to_insert.push(((state_to, cstr), (state_from, cstr, cost)));
                        }
                    }
                }

                let valid_state = to_insert.len() > 0;
                for v in to_insert {
                    updated = true;
                    let key = v.0;
                    let value = v.1;
                    let entry = self.table[idx].entry(key).or_insert((0, 0, f64::NEG_INFINITY));
                    if value.2 > entry.2 {
                        *entry = value;
                    }
                }
                
                if !valid_state && is_constrained {
                    valid_states[element.constraint_component as usize].remove(&state_to);
                }
            }

            if !updated {
                panic!("No valid states at time {} for element {:?}", idx, element);
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
            //
            // More Generally, if for all unfinished constraints, the choices are the same,
            // keep only the choice with the best value

            if is_constrained && element.last_of_constraint {
                finished_constraints[element.constraint_component as usize] = true;
                self.prune(idx, &finished_constraints);
            }
        }
        self.backtrack();
    }

    pub fn backtrack(&mut self) {
        let mut end_state: Option<(usize, usize)> = None;
        let mut best_cost = f64::NEG_INFINITY;
        for (key, val) in self.table[self.sequence.len()-1].iter() {
            let cost = val.2;
            if cost > best_cost {
                end_state = Some(*key);
                best_cost = cost;
            }
        }

        self.objective = best_cost;
        let mut current = end_state.unwrap();
        for idx in (0..self.solution.len()).rev() {
            self.solution[idx] = current.0;
            let from = self.table[idx].get(&current).unwrap();
            current = (from.0, from.1);
        }
    }

    pub fn parse_solution(&self) -> Array1<Array1<usize>> {
        self.sequence.parse_solution(&self.solution)
    }

    pub fn refresh_constraints(&mut self, proportion: f64) {
        self.sequence.recompute_constraints(proportion);
    }

    pub fn reorder(&mut self) {
        self.sequence.reorder();
    }
}
