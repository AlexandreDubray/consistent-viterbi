use ndarray::prelude::*;
use super::super::hmm::hmm::HMM;
use super::utils::SuperSequence;
use ndarray_stats::QuantileExt;

use super::Solver;

pub struct CPSolver<'a, const D: usize> {
    hmm: &'a HMM<D>,
    sequence: &'a SuperSequence<'a, D>,
    constraints: Vec<Vec<usize>>,
    best_obj: f64,
    best_sol: Array1<usize>
}

impl<'a, const D: usize> CPSolver<'a, D> {

    pub fn new(hmm: &'a HMM<D>, sequence: &'a SuperSequence<'a, D>) -> Self {
        // constraints is a vector of the consistency components. They are ordered by appearance in
        // the super-sequence. Each constraints is also ordered.
        let mut constraints: Vec<Vec<usize>> = (0..sequence.number_constraints()).map(|_| Vec::new()).collect();
        for t in 0..sequence.len() {
            if sequence[t].is_constrained() {
                let ucomp = sequence[t].constraint_component as usize;
                constraints[ucomp].push(t);
            }
        }
        Self {hmm, sequence, constraints, best_obj: f64::NEG_INFINITY, best_sol: Array1::from_elem(sequence.len(), 0)}
    }

    fn partial_viterbi(&mut self, array: &mut Array2<f64>, bt: &mut Array2<usize>, start: usize, end: usize, forced: &Array1<Option<usize>>) -> bool {
        if start == 0 {
            match forced[0] {
                None => {
                    let probs = self.hmm.init_probs(self.sequence[0].value);
                    array.row_mut(0).assign(&probs);
                },
                Some(state) => {
                    array.row_mut(0).fill(f64::NEG_INFINITY);
                    let p = self.hmm.init_prob(state, self.sequence[0].value);
                    array[[0, state]] = p;
                }
            };
        }
        let real_start = if start == 0 { 1 } else { start };
        for t in real_start..end {
            let mut upper_bound = f64::NEG_INFINITY;
            match forced[t] {
                None => {
                    for state_to in 0..self.hmm.nstates() {
                        if self.sequence[t].can_be_emited(self.hmm, state_to) {
                            let previous_prob = array.row(t-1);
                            let transitions = self.sequence[t].transitions(self.hmm, state_to);
                            let probs = &previous_prob + &transitions;
                            let state_from = probs.argmax().unwrap();
                            let arc_cost = self.sequence[t].arc_p(self.hmm, state_from, state_to);
                            let v = previous_prob[state_from] + arc_cost;
                            array[[t, state_to]] = v;
                            bt[[t, state_to]] = state_from;
                            if v > upper_bound {
                                upper_bound = v;
                            }
                        } else {
                            array[[t, state_to]] = f64::NEG_INFINITY;
                        }
                    }
                    if upper_bound < self.best_obj {
                        return false;
                    }
                },
                Some(state) => {
                    array.row_mut(t).fill(f64::NEG_INFINITY);
                    if !self.sequence[t].can_be_emited(self.hmm, state) {
                        return false;
                    }
                    let previous_prob = array.row(t-1);
                    let transitions = self.hmm.transitions_to(state);
                    let probs = &previous_prob + &transitions;
                    let state_from = probs.argmax().unwrap();
                    let v = previous_prob[state_from] + self.sequence[t].arc_p(self.hmm, state_from, state);
                    array[[t, state]] = v;
                    bt[[t, state]] = state_from;
                    upper_bound = v;
                    if upper_bound < self.best_obj {
                        return false;
                    }
                }
            };
        }
        true
    }

    fn backtrack(&mut self, array: &Array2<f64>, bt: &Array2<usize>) {
        let mut current = array.row(self.sequence.len() - 1).argmax().unwrap();
        let obj = array[[self.sequence.len()-1, current]];
        assert!(obj > self.best_obj);
        self.best_obj = obj;
        for t in (0..self.sequence.len()).rev() {
            self.best_sol[t] = current;
            current = bt[[t, current]];
        }
    }

    fn solve_r(&mut self, array: &mut Array2<f64>, bt: &mut Array2<usize>, comp: usize, forced: &mut Array1<Option<usize>>) {
        if comp >= self.constraints.len() {
            self.backtrack(array, bt);
        } else {
            for state in 0..self.hmm.nstates() {
                for pos in &self.constraints[comp] {
                    forced[*pos] = Some(state);
                }
                let start = self.constraints[comp][0];
                let end = if comp == self.constraints.len() - 1 { self.sequence.len() } else { self.constraints[comp + 1][0] };
                if self.partial_viterbi(array, bt, start, end, forced) {
                    self.solve_r(array, bt, comp+1, forced);
                }
            }

            for pos in &self.constraints[comp] {
                forced[*pos] = None;
            }
        }
    }
}

impl<'a, const D: usize> Solver for CPSolver<'a, D> {

    fn solve(&mut self) {
        let mut array = Array2::from_elem((self.sequence.len(), self.hmm.nstates()), 0.0);
        let mut bt = Array2::from_elem((self.sequence.len(), self.hmm.nstates()), 0);
        let mut forced: Array1<Option<usize>> = Array1::from_elem(self.sequence.len(), None);
        let e = if self.sequence.number_constraints() > 0 { self.constraints[0][0] } else { self.sequence.len() };
        self.partial_viterbi(&mut array, &mut bt, 0, e, &mut forced);
        self.solve_r(&mut array, &mut bt, 0, &mut forced);
    }

    fn get_solution(&self) -> &Array1<usize> {
        &self.best_sol
    }
    
    fn get_objective(&self) -> f64 { self.best_obj }

    fn get_name(&self) -> String { String::from("cp") }
}
