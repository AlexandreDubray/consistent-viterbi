use ndarray::prelude::*;
use super::super::hmm::hmm::HMM;
use super::utils::SuperSequence;
use ndarray_stats::QuantileExt;

use super::Solver;

pub struct CPSolver<'a, const D: usize> {
    hmm: &'a HMM<D>,
    sequence: &'a SuperSequence<'a, D>,
    constraints: Vec<Vec<usize>>,
    cstr_choices: Vec<Option<usize>>,
    best_obj: f64,
    best_sol: Array1<usize>,
    explored_nodes: usize
}

impl<'a, const D: usize> CPSolver<'a, D> {

    pub fn new(hmm: &'a HMM<D>, sequence: &'a SuperSequence<'a, D>) -> Self {
        let mut constraints: Vec<Vec<usize>> = (0..sequence.number_constraints()).map(|_| Vec::new()).collect();
        for t in 0..sequence.len() {
            if sequence[t].is_constrained() {
                let ucomp = sequence[t].constraint_component as usize;
                constraints[ucomp].push(t);
            }
        }
        let cstr_choices: Vec<Option<usize>> = (0..sequence.number_constraints()).map(|_| None).collect();
        Self {hmm, sequence, constraints, cstr_choices, best_obj: f64::NEG_INFINITY, best_sol: Array1::from_elem(sequence.len(), 0), explored_nodes: 0}
    }

    pub fn viterbi_from(&mut self, array: &mut Array2<f64>, bt: &mut Array2<usize>, from: usize, node: usize) {
        array.row_mut(from).fill(f64::NEG_INFINITY);
        array[[from, node]] = 0.0;
        if from != 0 {
            let prev = array.row(from-1);
            let transitions = self.sequence[from].transitions(self.hmm, node);
            let p = &prev + &transitions;
            let sfrom = p.argmax().unwrap();
            bt[[from, node]] = sfrom;
        }

        let mut t = from + 1;
        while t < self.sequence.len() && !(self.sequence[t].is_constrained() && self.cstr_choices[self.sequence[t].constraint_component as usize].is_some()) {
            for state in 0..self.hmm.nstates() {
                let prev_probs = array.row(t-1);
                let transitions = self.sequence[t].transitions(self.hmm, state);
                let probs = &prev_probs + &transitions;
                let state_from = probs.argmax().unwrap();
                let arc_cost = self.sequence[t].arc_p(self.hmm, state_from, state);
                let v = prev_probs[state_from] + arc_cost;
                array[[t, state]] = v;
                bt[[t, state]] = state_from;
            }
            t += 1;
        }
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

    fn backtrack(&mut self, array: &Array2<f64>, bt: &Array2<usize>, obj: f64) {
        let mut current = array.row(self.sequence.len() - 1).argmax().unwrap();
        assert!(obj > self.best_obj);
        self.best_obj = obj;
        for t in (0..self.sequence.len()).rev() {
            self.best_sol[t] = current;
            current = bt[[t, current]];
        }
    }

    fn solve_r(&mut self, array: &mut Array2<f64>, bt: &mut Array2<usize>, comp: usize) {
        for state in 0..self.hmm.nstates() {
            self.explored_nodes += 1;
            self.cstr_choices[comp] = Some(state);
            for idx in 0..self.constraints[comp].len() {
                let pos = self.constraints[comp][idx];
                self.viterbi_from(array, bt, pos, state);
            }
            let mut ub = 0.0;
            for cid in 0..(comp+1) {
                let state = self.cstr_choices[cid].unwrap();
                for idx in &self.constraints[cid] {
                    let t = *idx;
                    if t == 0 {
                        ub += self.sequence[0].arc_p(self.hmm, 0, state);
                    } else {
                        let sf = bt[[t, state]];
                        let arc_cost = self.sequence[t].arc_p(self.hmm, sf, state);
                        ub += array[[t-1, sf]] + arc_cost;
                    }
                }
            }
            if ub >= self.best_obj {
                if comp < self.constraints.len() {
                    self.solve_r(array, bt, comp+1);
                } else {
                    self.backtrack(array, bt, ub);
                }
            }
        }
        self.cstr_choices[comp] = None;
    }

    pub fn get_explored_nodes(&self) -> usize { self.explored_nodes }
}

impl<'a, const D: usize> Solver for CPSolver<'a, D> {

    fn solve(&mut self) {
        let mut array = Array2::from_elem((self.sequence.len(), self.hmm.nstates()), 0.0);
        let mut bt = Array2::from_elem((self.sequence.len(), self.hmm.nstates()), 0);
        self.init_viterbi(&mut array, &mut bt);
        self.solve_r(&mut array, &mut bt, 0);
    }

    fn get_solution(&self) -> &Array1<usize> {
        &self.best_sol
    }
    
    fn get_objective(&self) -> f64 { self.best_obj }

    fn get_name(&self) -> String { String::from("cp") }
}
