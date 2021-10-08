use ndarray::{Array2, Array1, s};
use ndarray::ArrayView1;

pub struct HMM {
    a: Array2<f64>,
    b: Array2<f64>,
    pi: Array1<f64>,
}

impl HMM {

    pub fn new(a: Array2<f64>, b: Array2<f64>, pi: Array1<f64>) -> Self {
        //TODO: checks
        Self{ a, b, pi}
    }

    pub fn nstates(&self) -> usize {
        self.a.nrows()
    }

    pub fn init_prob(&self, state: usize) -> f64 {
        self.pi[state]
    }

    pub fn init_prob_obs(&self, obs: usize) -> Array1<f64> {
        self.pi.clone() + self.b.slice(s![.., obs])
    }

    pub fn transitions_from(&self, state_from: usize) -> ArrayView1<f64> {
        self.a.slice(s![state_from, ..])
    }

    pub fn transitions_to(&self, state_to: usize) -> ArrayView1<f64> {
        self.a.slice(s![.., state_to])
    }

    pub fn emit_prob(&self, state: usize, obs: usize) -> f64 {
        self.b[[state, obs]]
    }

    pub fn can_emit(&self, state: usize, obs: usize, t: usize) -> bool {
        if t == 0 {
            self.pi[state] + self.emit_prob(state, obs) > f64::NEG_INFINITY
        } else {
            self.emit_prob(state, obs) > f64::NEG_INFINITY
        }
    }
}
