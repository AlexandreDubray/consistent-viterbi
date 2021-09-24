use ndarray::{Array2, Array1};
use ndarray_stats::QuantileExt;
use super::hmm::HMM;


pub struct Viterbi {
    hmm: HMM,
    viterbi_array: Array2<f64>,
    viterbi_bt: Array2<usize>
}

impl Viterbi {

    pub fn new(hmm: HMM, max_seq_size: usize) -> Self {
        let viterbi_array: Array2<f64> = Array2::zeros((max_seq_size, hmm.nstates()));
        let viterbi_bt: Array2<usize> = Array2::zeros((max_seq_size, hmm.nstates()));
        Self {hmm, viterbi_array, viterbi_bt}
    }

    pub fn solve(&mut self, sequence: &Array1<usize>) -> Array1<usize> {
        // Reset the log probs
        self.viterbi_array.row_mut(0).assign(&self.hmm.init_prob(sequence[0]));

        for t in 1..sequence.len() {
            for state_to in 0..self.hmm.nstates() {
                let emit_prob = self.hmm.emit_prob(state_to, sequence[t]);
                if emit_prob > f64::NEG_INFINITY {
                    let previous_prob = self.viterbi_array.row(t-1);
                    let transitions = self.hmm.transition(state_to);
                    let probs = &previous_prob + &transitions;
                    let state_from = probs.argmax().unwrap();
                    self.viterbi_array[[t, state_to]] = probs[state_from] + emit_prob;
                    self.viterbi_bt[[t, state_to]] = state_from;
                } else {
                    self.viterbi_array[[t, state_to]] = f64::NEG_INFINITY;
                }
            }
        }

        let mut end_state = self.viterbi_array.row(sequence.len()-1).argmax().unwrap();
        let mut predicted = Array1::zeros(sequence.len());
        predicted[sequence.len()-1] = end_state;
        for t in (0..sequence.len()-1).rev() {
            end_state = self.viterbi_bt[[t+1, end_state]];
            predicted[t] = end_state;
        }
        predicted
    }
}
