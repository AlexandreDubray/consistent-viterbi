use ndarray::{Array1, Array2};
use super::super::hmm::hmm::HMM;
use ndarray_stats::QuantileExt;

pub fn decode<const D: usize>(sequence: &Vec<[usize; D]>, hmm: &HMM<D>) -> Array1<usize> {
    let mut array = Array2::from_elem((sequence.len(), hmm.nstates()), 0.0);
    let mut bt = Array2::from_elem((sequence.len(), hmm.nstates()), 0);

    for t in 1..sequence.len() {
        for state_to in 0..hmm.nstates() {
            let emit_prob = hmm.emit_prob(state_to, sequence[t]);
            if emit_prob > f64::NEG_INFINITY {
                let previous_prob = array.row(t-1);
                let transitions = hmm.transitions_to(state_to);
                let probs = &previous_prob + &transitions;
                let state_from = probs.argmax().unwrap();
                array[[t, state_to]] = probs[state_from] + emit_prob;
                bt[[t, state_to]] = state_from;
            } else {
                array[[t, state_to]] = f64::NEG_INFINITY;
            }
        }
    }
    let mut end_state = array.row(sequence.len()-1).argmax().unwrap();
    let mut predicted = Array1::zeros(sequence.len());
    predicted[sequence.len()-1] = end_state;
    for t in (0..sequence.len()-1).rev() {
        end_state = bt[[t+1, end_state]];
        predicted[t] = end_state;
    }
    predicted
}
