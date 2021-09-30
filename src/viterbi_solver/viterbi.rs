use ndarray::{Array2, Array1};
use ndarray_stats::QuantileExt;
use super::hmm::HMM;
use std::collections::HashMap;


pub struct Viterbi<'a> {
    hmm: &'a HMM,
    viterbi_array: Array2<f64>,
    viterbi_bt: Array2<usize>
}

impl<'b> Viterbi<'b> {

    pub fn new(hmm: &'b HMM, max_seq_size: usize) -> Self {
        let viterbi_array: Array2<f64> = Array2::zeros((max_seq_size, hmm.nstates()));
        let viterbi_bt: Array2<usize> = Array2::zeros((max_seq_size, hmm.nstates()));
        Self {hmm, viterbi_array, viterbi_bt}
    }

    fn backtrack(&self, seq_len: usize) -> Array1<usize> {
        let mut end_state = self.viterbi_array.row(seq_len-1).argmax().unwrap();
        let mut predicted = Array1::zeros(seq_len);
        predicted[seq_len-1] = end_state;
        for t in (0..seq_len-1).rev() {
            end_state = self.viterbi_bt[[t+1, end_state]];
            predicted[t] = end_state;
        }
        predicted
    }

    fn update_state_from(&mut self, state: usize, t: usize, emit_prob: f64) {
        if emit_prob > f64::NEG_INFINITY {
            let previous_prob = self.viterbi_array.row(t-1);
            let transitions = self.hmm.transition(state);
            let probs = &previous_prob + &transitions;
            let state_from = probs.argmax().unwrap();
            self.viterbi_array[[t, state]] = probs[state_from] + emit_prob;
            self.viterbi_bt[[t, state]] = state_from;
        } else {
            self.viterbi_array[[t, state]] = f64::NEG_INFINITY;
        }
    }

    pub fn solve(&mut self, sequence: &Array1<usize>) -> Array1<usize> {
        // Reset the log probs
        self.viterbi_array.row_mut(0).assign(&self.hmm.init_prob(sequence[0]));

        for t in 1..sequence.len() {
            for state_to in 0..self.hmm.nstates() {
                let emit_prob = self.hmm.emit_prob(state_to, sequence[t]);
                self.update_state_from(state_to, t, emit_prob);
            }
        }
        self.backtrack(sequence.len())
    }

    pub fn solve_must_visit(&mut self, sequence: &Array1<usize>, must_visit: &HashMap<usize, usize>) -> Array1<usize> {
        self.viterbi_array.row_mut(0).assign(&self.hmm.init_prob(sequence[0]));

        for t in 1..sequence.len() {
            let range = match must_visit.get(&t) {
                Some(x) => *x..(*x+1),
                None => 0..self.hmm.nstates()
            };
            for state_to in range {
                let emit_prob = self.hmm.emit_prob(state_to, sequence[t]);
                self.update_state_from(state_to, t, emit_prob);
            }
        }
        self.backtrack(sequence.len())
    }
}
