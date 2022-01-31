use std::collections::HashMap;
use ndarray::prelude::*;
use super::super::hmm::hmm::HMM;
use super::utils::SuperSequence;
use ndarray_stats::QuantileExt;

use super::Solver;

pub struct CPSolver<'a, const D: usize> {
    hmm: &'a HMM<D>,
    sequence: &'a SuperSequence<'a, D>,
    constraints: Vec<Vec<usize>>,
    previous_cstr: HashMap<usize, Option<usize>>,
    cstr_choices: Vec<Option<usize>>,
    best_obj: f64,
    best_sol: Array1<usize>,
}

impl<'a, const D: usize> CPSolver<'a, D> {

    pub fn new(hmm: &'a HMM<D>, sequence: &'a SuperSequence<'a, D>) -> Self {
        let mut constraints: Vec<Vec<usize>> = (0..sequence.number_constraints()).map(|_| Vec::new()).collect();
        let mut previous_cstr: HashMap<usize, Option<usize>> = HashMap::new();
        let mut last_cstr: Option<usize> = None;
        for t in 0..sequence.len() {
            if sequence[t].is_constrained() {
                let ucomp = sequence[t].constraint_component as usize;
                constraints[ucomp].push(t);
                previous_cstr.insert(t, last_cstr);
                last_cstr = Some(t);
            }
        }
        let cstr_choices: Vec<Option<usize>> = (0..sequence.number_constraints()).map(|_| None).collect();
        Self {hmm, sequence, constraints, previous_cstr, cstr_choices, best_obj: f64::NEG_INFINITY, best_sol: Array1::from_elem(sequence.len(), 0)}
    }

    fn init_viterbi(&mut self, array: &mut Array2<f64>, bt: &mut Array2<usize>) {
        let mut t = 0;
        while t < self.sequence.len() && !self.sequence[t].is_constrained() {
            if t == 0 {
                let probs = self.hmm.init_probs(self.sequence[t].value);
                array.row_mut(t).assign(&probs);
            } else {
                for state_to in 0..self.hmm.nstates() {
                    let previous_prob = array.row(t-1);
                    let transitions = self.sequence[t].transitions(self.hmm, state_to);
                    let probs = &previous_prob + &transitions;
                    let state_from = probs.argmax().unwrap();
                    let arc_cost = self.sequence[t].arc_p(self.hmm, state_from, state_to);
                    let v = previous_prob[state_from] + arc_cost;
                    array[[t, state_to]] = v;
                    bt[[t, state_to]] = state_from;
                }
            }
            t += 1;
        }
    }

    fn partial_viterbi(&mut self, array: &mut Array2<f64>, bt: &mut Array2<usize>, start: usize, ub: f64) -> f64 {
        let mut new_ub = ub;
        assert!(self.sequence[start].is_constrained());
        let cid = self.sequence[start].constraint_component as usize;
        let state = self.cstr_choices[cid].unwrap();
        array.row_mut(start).fill(f64::NEG_INFINITY);
        array[[start, state]] = 0.0;
        // If the previous "part" was computed, update the upper bound
        let prev_computed = match self.previous_cstr.get(&start).unwrap() {
            None => true,
            Some(t) => {
                let cid = self.sequence[*t].constraint_component as usize;
                self.cstr_choices[cid].is_some()
            }
        };
        if prev_computed {
            if start == 0 {
                new_ub += self.sequence[start].arc_p(self.hmm, 0, state);
            } else {
                let prev_probs = array.row(start-1);
                let transitions = self.sequence[start].transitions(self.hmm, state);
                let probs = &prev_probs + &transitions;
                let state_from = probs.argmax().unwrap();
                new_ub += prev_probs[state_from] + self.sequence[start].arc_p(self.hmm, state_from, state);
                bt[[start, state]] = state_from;
            }
        }

        let mut t = start + 1;
        while t < self.sequence.len() && !self.sequence[t].is_constrained() {
            for state_to in 0..self.hmm.nstates() {
                let previous_prob = array.row(t-1);
                let transitions = self.sequence[t].transitions(self.hmm, state_to);
                let probs = &previous_prob + &transitions;
                let state_from = probs.argmax().unwrap();
                let arc_cost = self.sequence[t].arc_p(self.hmm, state_from, state_to);
                let v = previous_prob[state_from] + arc_cost;
                array[[t, state_to]] = v;
                bt[[t, state_to]] = state_from;
            }
            t += 1;
        }
        if t < self.sequence.len() {
            let ocid = self.sequence[t].constraint_component as usize;
            if ocid != cid {
                if let Some(state) = self.cstr_choices[ocid] {
                    let prev_probs = array.row(t-1);
                    let transitions = self.sequence[t].transitions(self.hmm, state);
                    let probs = &prev_probs + &transitions;
                    let state_from = probs.argmax().unwrap();
                    new_ub += prev_probs[state_from] + self.sequence[t].arc_p(self.hmm, state_from, state);
                    bt[[t, state]] = state_from;
                }
            }
        } else {
            new_ub += array.row(self.sequence.len()-1).max().unwrap();
        }
        new_ub
    }

    fn backtrack(&mut self, array: &Array2<f64>, bt: &Array2<usize>, obj: f64) {
        let mut current = array.row(self.sequence.len() - 1).argmax().unwrap();
        assert!(obj > self.best_obj);
        self.best_obj = obj;
        for t in (0..self.sequence.len()).rev() {
            self.best_sol[t] = current;
            current = bt[[t, current]];
        }
    }

    fn solve_r(&mut self, array: &mut Array2<f64>, bt: &mut Array2<usize>, comp: usize, ub: f64) {
        if comp >= self.constraints.len() {
            self.backtrack(array, bt, ub);
        } else {
            let positions: Vec<usize> = (0..self.constraints[comp].len()).map(|i| self.constraints[comp][i]).collect();
            for state in 0..self.hmm.nstates() {
                let mut new_ub = ub;
                let mut should_bt = false;
                self.cstr_choices[comp] = Some(state);
                for pos in &positions {
                    new_ub = self.partial_viterbi(array, bt, *pos, new_ub);
                    if new_ub <= self.best_obj {
                        should_bt = true;
                        break;
                    }
                }
                if !should_bt {
                    self.solve_r(array, bt, comp+1, new_ub);
                }
            }
            self.cstr_choices[comp] = None;
        }
    }

}

impl<'a, const D: usize> Solver for CPSolver<'a, D> {

    fn solve(&mut self) {
        let mut array = Array2::from_elem((self.sequence.len(), self.hmm.nstates()), 0.0);
        let mut bt = Array2::from_elem((self.sequence.len(), self.hmm.nstates()), 0);
        self.init_viterbi(&mut array, &mut bt);
        self.solve_r(&mut array, &mut bt, 0, 0.0);
    }

    fn get_solution(&self) -> &Array1<usize> {
        &self.best_sol
    }
    
    fn get_objective(&self) -> f64 { self.best_obj }

    fn get_name(&self) -> String { String::from("cp") }
}
