use ndarray::{Array2, Array1};
use ndarray_stats::QuantileExt;
use std::collections::HashMap;
use std::collections::HashSet;
use std::ops::Range;

use super::hmm::HMM;
use super::constraints::Constraints;

enum LagrangianResult {
    Optimum,
    MaxIteration,
    SlowConvergence
}

pub struct Lagrangian {
    hmm: HMM,
    sequences: Vec<Array1<usize>>,
    viterbi_array: Array2<f64>,
    viterbi_bt: Array2<usize>,
    lambdas: HashMap<(usize, usize), Array1<f64>>,
    constraints: Constraints,
    solutions: Vec<Array1<usize>>,
    best_obj: f64,
    cur_solutions: Vec<Array1<usize>>,
    cur_log_probs: Array1<f64>,
    must_visit_constraints: Vec<HashMap<usize, usize>>
}

impl Lagrangian {

    pub fn new(hmm: HMM, sequences: Vec<Array1<usize>>, max_seq_size: usize, constraints: Constraints) -> Self {
        // TODO add checks here   
        let viterbi_array: Array2<f64> = Array2::zeros((max_seq_size, hmm.nstates()));
        let viterbi_bt: Array2<usize> = Array2::zeros((max_seq_size, hmm.nstates()));

        let mut lambdas: HashMap<(usize, usize), Array1<f64>> = HashMap::new();

        for component in &constraints.components {
            for (seq, t) in component {
                lambdas.insert((*seq, *t), Array1::zeros(hmm.nstates()));
            }
        }

        let mut solutions: Vec<Array1<usize>> = Vec::with_capacity(sequences.len());
        let mut cur_solutions: Vec<Array1<usize>> = Vec::with_capacity(sequences.len());
        let cur_log_probs = Array1::zeros(sequences.len());
        let best_obj = 0.0;
        let mut must_visit_constraints: Vec<HashMap<usize, usize>> = Vec::with_capacity(sequences.len());
        for i in 0..sequences.len() {
            solutions.push(Array1::zeros(1));
            cur_solutions.push(Array1::zeros(1));
            must_visit_constraints.push(HashMap::new());
        }

        Self {hmm, sequences, viterbi_array, viterbi_bt, lambdas, constraints, solutions, best_obj, cur_solutions, cur_log_probs, must_visit_constraints}
    }

    fn get_lambdas(&self, seq_id: usize, t: usize, state: usize) -> f64 {
        match self.lambdas.get(&(seq_id, t)) {
            Some(x) => x[state],
            None => 0.0
        }
    }

    fn get_state_range(&self, t: usize, must_visit: &HashMap<usize, usize>) -> Range<usize> {
        match must_visit.get(&t) {
            Some(x) => *x..(*x+1),
            None => 0..self.hmm.nstates()
        }
    }

    fn solve_viterbi(&mut self, seq_id: usize) -> (f64, Array1<usize>) {
        // Reset the log probs
        self.viterbi_array.fill(f64::NEG_INFINITY);
        let sequence = &self.sequences[seq_id];
        let must_visit = &self.must_visit_constraints[seq_id];
        match must_visit.get(&0) {
            Some(x) => self.viterbi_array[[0, *x]] = self.hmm.init_prob(*x) + self.hmm.emit_prob(*x, sequence[0]),
            None => self.viterbi_array.row_mut(0).assign(&self.hmm.init_prob_obs(sequence[0]))
        };

        for t in 1..sequence.len() {
            for state_to in self.get_state_range(t, must_visit) {
                let emit_prob = self.hmm.emit_prob(state_to, sequence[t]);
                if emit_prob > f64::NEG_INFINITY {
                    let previous_prob = self.viterbi_array.row(t-1);
                    let transitions = self.hmm.transitions_to(state_to);
                    let probs = &previous_prob + &transitions;
                    let state_from = probs.argmax().unwrap();
                    self.viterbi_array[[t, state_to]] = probs[state_from] + emit_prob;
                    self.viterbi_bt[[t, state_to]] = state_from;
                }
            }
        }

        let mut end_state = self.viterbi_array.row(sequence.len()-1).argmax().unwrap();
        let end_prob = self.viterbi_array[[sequence.len()-1, end_state]];
        let mut predicted = Array1::zeros(sequence.len());
        predicted[sequence.len()-1] = end_state;
        for t in (0..sequence.len()-1).rev() {
            end_state = self.viterbi_bt[[t+1, end_state]];
            predicted[t] = end_state;
        }
        (end_prob, predicted)
    }

    fn check_constraints(&self) -> bool {
        for component in &self.constraints.components {
            for i in 0..component.len() - 1 {
                let (s1, t1) = component[i];
                let (s2, t2) = component[i+1];
                if self.cur_solutions[s1][t1] != self.cur_solutions[s2][t2] {
                    return false;
                }
            }
        }
        true
    }

    fn update_lambas(&mut self, alpha: f64) {
        for component in &self.constraints.components {
            for i in 0..component.len() - 1 {
                let (s1, t1) = component[i];
                let (s2, t2) = component[i+1];
                let p1 = self.cur_solutions[s1][t1];
                let p2 = self.cur_solutions[s2][t2];
                if p1  != p2 {
                    let l1 = self.lambdas.get_mut(&(s1, t1)).unwrap();
                    l1[p1] += alpha;
                    let l2 = self.lambdas.get_mut(&(s2, t2)).unwrap();
                    l2[p2] -= alpha;
                }
            }
        }
    }

    fn solve_lagrangian(&mut self, max_iter: usize, min_delta: f64, ids_to_solve: &HashSet<usize>) -> LagrangianResult {
        let mut last_objective: Option<f64> = None;
        for iter in 1..max_iter+1 {
            println!("Iteration {}", iter);
            let mut objective = 0.0;
            for idx in 0..self.sequences.len() {
                if idx == 0 || ids_to_solve.contains(&idx) {
                    let (log_prob, sol) = self.solve_viterbi(idx); 
                    self.cur_solutions[idx] = sol;
                    self.cur_log_probs[idx] = log_prob;
                    objective += log_prob;
                } else if idx > 0 {
                    objective += self.cur_log_probs[idx];
                }
            }

            let optimal = self.check_constraints();

            if optimal {
                return LagrangianResult::Optimum;
            }

            match last_objective {
                Some(x) => {
                    let diff = x - objective;
                    if diff.abs() < min_delta {
                        return LagrangianResult::SlowConvergence;
                    }
                },
                None => ()
            }
            last_objective = Some(objective);
            let alpha = 1.0 / iter as f64;
            self.update_lambas(alpha);
        }
        LagrangianResult::MaxIteration
    }

    fn _dfs(&mut self, comp_id: usize) {
        println!("dfs at comp_id {}", comp_id);
        if comp_id == self.constraints.components.len() {
            let obj: f64 = self.cur_log_probs.iter().sum();
            if obj > self.best_obj {
                self.best_obj = obj;
                self.solutions = self.cur_solutions.clone();
            }
            return
        }
        let mut ids_to_solve: HashSet<usize> = HashSet::new();
        for (seq_id, t) in self.constraints.get_component(comp_id) {
            ids_to_solve.insert(*seq_id);
        }
        for state in 0..self.hmm.nstates() {
            let component = self.constraints.get_component(comp_id);
            for (seq_id, t) in component {
                *self.must_visit_constraints[*seq_id].entry(*t).or_insert(state) = state;
            }
            match self.solve_lagrangian(100, 0.001, &ids_to_solve) {
                LagrangianResult::Optimum => {
                    println!("Optimum found for lagrangian relaxation");
                    let obj: f64 = self.cur_log_probs.iter().sum();
                    if obj > self.best_obj {
                        self.best_obj = obj;
                        self.solutions = self.cur_solutions.clone();
                    }
                }
                _ => {
                    let obj: f64 = self.cur_log_probs.iter().sum();
                    if obj > self.best_obj {
                        self._dfs(comp_id + 1);
                    }
                }
            };
        }
    }

    pub fn solve_bb(&mut self) {
        self._dfs(0);
    }

    pub fn solve(&mut self) {
        let ids_to_solve: HashSet<usize> = Array1::from_iter(0..self.sequences.len()).iter().cloned().collect();
        match self.solve_lagrangian(100, 0.001, &ids_to_solve) {
            LagrangianResult::Optimum => {
                println!("Optimum found for lagrangian relaxation");
            },
            LagrangianResult::MaxIteration => {
                println!("Maximum number of iterations reached");
            },
            LagrangianResult::SlowConvergence => {
                println!("Converging slowly");
            }
        };
    }

}
