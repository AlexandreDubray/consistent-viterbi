use ndarray::{Array2, Array1, s};
use ndarray::ArrayView1;
use ndarray_stats::QuantileExt;

pub struct HMM {
    pub a: Array2<f64>,
    pub b: Array2<f64>,
    pub pi: Array1<f64>,
    viterbi_array: Array2<f64>,
    viterbi_bt: Array2<usize>
}

impl HMM {

    pub fn new(sequences: &Array1<Array1<usize>>, tags: &Array1<Array1<usize>>, nstates: usize, nobs: usize) -> Self {
        let mut a = Array2::from_elem((nstates, nstates), 0.0);
        let mut b = Array2::from_elem((nstates, nobs), 0.0);
        let mut pi = Array1::from_elem(nstates, 0.0);

        let mut seen_states = Array1::from_elem(nstates, 0.0);
        let mut end_states = Array1::from_elem(nstates, 0.0);

        let mut max_seq_size = 0;

        for i in 0..sequences.len() {
            let sequence = &sequences[i];
            let tag = &tags[i];
            max_seq_size = max_seq_size.max(sequence.len());

            pi[tag[0]] += 1.0;
            for t in 0..sequence.len() - 1 {
                b[[tag[t], sequence[t]]] += 1.0;
                a[[tag[t], tag[t+1]]] += 1.0;
                seen_states[tag[t]] += 1.0;
            }
            b[[tag[sequence.len()-1], sequence[sequence.len()-1]]] += 1.0;
            seen_states[tag[sequence.len()-1]] += 1.0;
            end_states[tag[sequence.len()-1]] += 1.0;
        }

        let log_mapping = |x: &mut f64| {
            if *x == 0.0 || x.is_nan() {
                *x = f64::NEG_INFINITY;
            } else {
                *x = x.log(10.0);
            }
        };

        for state in 0..nstates {
            let mut a_row = a.row_mut(state);
            if seen_states[state] != end_states[state] {
                a_row /= seen_states[state] - end_states[state];
            } else {
                a_row.fill(0.0); 
            }
            a_row.map_inplace(log_mapping);
            pi[state] = if pi[state] == 0.0 { f64::NEG_INFINITY } else { pi[state] / sequences.len() as f64 };
            let mut b_row = b.row_mut(state);
            b_row /= seen_states[state];
            b_row.map_inplace(log_mapping);
        }

        let viterbi_array: Array2<f64> = Array2::zeros((max_seq_size, nstates));
        let viterbi_bt: Array2<usize> = Array2::zeros((max_seq_size, nstates));

        Self {a, b, pi, viterbi_array, viterbi_bt}
    }

    pub fn nstates(&self) -> usize {
        self.a.nrows()
    }

    pub fn init_prob(&self, state: usize, obs: usize) -> f64 {
        self.pi[state] + self.b[[state, obs]]
    }

    pub fn transition_prob(&self, state_from: usize, state_to: usize, obs: usize) -> f64 {
        self.a[[state_from, state_to]] + self.b[[state_to, obs]]
    }

    pub fn transitions_to(&self, state_to: usize) -> ArrayView1<f64> {
        self.a.slice(s![.., state_to])
    }

    pub fn emit_prob(&self, state: usize, obs: usize) -> f64 {
        self.b[[state, obs]]
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
            let transitions = self.transitions_to(state);
            let probs = &previous_prob + &transitions;
            let state_from = probs.argmax().unwrap();
            self.viterbi_array[[t, state]] = probs[state_from] + emit_prob;
            self.viterbi_bt[[t, state]] = state_from;
        } else {
            self.viterbi_array[[t, state]] = f64::NEG_INFINITY;
        }
    }

    pub fn decode(&mut self, sequence: &Array1<usize>) -> Array1<usize> {
        // Reset the log probs
        for state in 0..self.nstates() {
            self.viterbi_array[[0, state]] = self.init_prob(state, sequence[0]);
        }

        for t in 1..sequence.len() {
            for state_to in 0..self.nstates() {
                let emit_prob = self.emit_prob(state_to, sequence[t]);
                self.update_state_from(state_to, t, emit_prob);
            }
        }
        self.backtrack(sequence.len())
    }

}
