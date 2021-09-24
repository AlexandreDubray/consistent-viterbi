use ndarray::{Array2, Array1, s};
use ndarray::ArrayView1;

pub struct HMM {
    a: Array2<f64>,
    b: Array2<f64>,
    pi: Array1<f64>
}

impl HMM {

    pub fn new(a: Array2<f64>, b: Array2<f64>, pi: Array1<f64>) -> Self {
        //TODO: checks
        Self{ a, b, pi}
    }

    #[inline(always)]
    pub fn nstates(&self) -> usize {
        self.a.nrows()
    }

    #[inline(always)]
    pub fn nobs(&self) -> usize {
        self.b.ncols()
    }

    #[inline(always)]
    pub fn init_prob(&self, obs: usize) -> Array1<f64> {
        self.pi.clone() + self.b.slice(s![.., obs])
    }

    #[inline(always)]
    pub fn single_init_prob(&self, state: usize, obs: usize) -> f64 {
        self.pi[state] + self.b[[state, obs]]
    }

    #[inline(always)]
    pub fn transition(&self, state_to: usize) -> ArrayView1<f64> {
        self.a.slice(s![.., state_to])
    }

    #[inline(always)]
    pub fn emit_prob(&self, state: usize, obs: usize) -> f64 {
        self.b[[state, obs]]
    }

}
