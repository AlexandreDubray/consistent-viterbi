use ndarray::prelude::*;
use rand::prelude::*;
use std::time::Instant;

pub struct HMM<const D: usize> {
    pub a: Array2<f64>,
    pub b: Array1<Array<f64, IxDyn>>,
    pub pi: Array1<f64>,
}

impl<const D: usize> HMM<D> {

    pub fn new(nstates: usize, bdims: [usize; D]) -> Self {
        let mut rng = thread_rng();
        let mut a = Array2::from_shape_fn((nstates, nstates), |_| rng.gen::<f64>());
        let mut b = Array1::from_shape_fn(nstates, |_| Array::from_shape_fn(&bdims[..], |_| rng.gen::<f64>()));
        let mut pi = Array1::from_shape_fn(nstates, |_| rng.gen::<f64>());
        for state in 0..nstates {
            let mut ra = a.row_mut(state);
            let s = ra.sum();
            ra /= s;
            let mb = &b[state];
            let s = mb.sum();
            b[state] = mb / s;
        }
        let s = pi.sum();
        pi /= s;
        Self { a, b, pi }
    }

    pub fn train_supervised(&mut self, sequences: &Vec<Vec<[usize; D]>>, tags: &Vec<Vec<usize>>) {
        let nstates = self.a.nrows();
        let mut seen_states = Array1::<f64>::zeros(nstates);
        let mut end_states = Array1::<f64>::zeros(nstates);

        for i in 0..sequences.len() {
            let sequence = &sequences[i];
            let tag = &tags[i];

            self.pi[tag[0]] += 1.0;
            for t in 0..sequence.len() - 1 {
                self.b[tag[t]][&sequence[t][..]] += 1.0;
                self.a[[tag[t], tag[t+1]]] += 1.0;
                seen_states[tag[t]] += 1.0;
            }
            self.b[tag[sequence.len()-1]][&sequence[sequence.len()-1][..]] += 1.0;
            seen_states[tag[sequence.len()-1]] += 1.0;
            end_states[tag[sequence.len()-1]] += 1.0;
        }

        for state in 0..nstates {
            let mut a_row = self.a.row_mut(state);
            if seen_states[state] != end_states[state] {
                a_row /= seen_states[state] - end_states[state];
            } else {
                a_row.fill(0.0); 
            }
            self.pi[state] /= sequences.len() as f64;
            self.b[state] /= seen_states[state];
        }
    }

    fn check_row_sum(&self, row: ArrayView1<f64>) -> bool {
        let d = row.sum() - 1.0;
        d.abs() <= 0.0001
    }

    pub fn train_semi_supervised(&mut self, sequences: &Vec<Vec<[usize; D]>>, tags: &Vec<Vec<Option<usize>>>, prior_transmat: Option<Array2<f64>>, max_iter: usize, tol: f64) {
        let nstates = self.a.nrows();

        match prior_transmat {
            Some(mat) => self.a.assign(&mat),
            None => ()
        };
        let mut a_t = self.a.t();

        let mut pi_num = Array1::from_elem(nstates, 0.0);
        let mut omega_num = Array1::from_elem(nstates, 0.0);
        let mut a_num = Array2::from_elem((nstates, nstates), 0.0);
        let mut a_den = Array1::from_elem(nstates, 0.0);
        let mut b_den = Array1::from_elem(nstates, 0.0);
        let mut b_num = Array1::from_shape_fn(nstates, |i| Array::from_elem(self.b[i].raw_dim(), 0.0));

        let mut max_seq_size = 0;
        for sequence in sequences {
            max_seq_size = max_seq_size.max(sequence.len());
        }

        let mut alphas = Array2::from_elem((max_seq_size, nstates), 0.0);
        let mut betas = Array2::from_elem((max_seq_size, nstates), 0.0);
        let mut gammas = Array2::from_elem((max_seq_size, nstates), 0.0);
        let mut xis = Array3::from_elem((max_seq_size, nstates, nstates), 0.0);

        for iter in 0..max_iter {
            println!("Iteration {}/{}", iter+1, max_iter);
            let start_it = Instant::now();
            pi_num.fill(0.0);
            omega_num.fill(0.0);
            a_num.fill(0.0);
            a_den.fill(0.0);
            b_den.fill(0.0);
            for state in 0..nstates {
                b_num[state].fill(0.0);
            }

            for seq_id in 0..sequences.len() {
                let sequence = &sequences[seq_id];
                for t in 0..sequence.len() {
                    let beta_index = sequence.len() - 1 - t;
                    let update_t = match tags[seq_id][t] {
                        Some(state) => {
                            alphas.row_mut(t).fill(0.0);
                            alphas[[t, state]] = 1.0;
                            betas.row_mut(t).fill(0.0);
                            betas[[t, state]] = 1.0;
                            false
                        },
                        None => true
                    };
                    let update_beta_t = match tags[seq_id][beta_index] {
                        Some(state) => {
                            alphas.row_mut(beta_index).fill(0.0);
                            alphas[[beta_index, state]] = 1.0;
                            betas.row_mut(beta_index).fill(0.0);
                            betas[[beta_index, state]] = 1.0;
                            false
                        },
                        None => true
                    };
                    if t == 0 {
                        if update_t {
                            let bmul = self.b.map(|r| r[&sequence[t][..]]);
                            let new_values = &self.pi * &bmul;
                            alphas.row_mut(t).assign(&new_values);
                        }
                        if update_beta_t {
                            betas.row_mut(beta_index).fill(1.0);
                        }
                    } else {
                        if update_t {
                            let bmul = self.b.map(|r| r[&sequence[t][..]]);
                            let new_values = alphas.row(t-1).dot(&self.a)*&bmul;
                            alphas.row_mut(t).assign(&new_values);
                        }
                        if update_beta_t {
                            let bmul = self.b.map(|r| r[&sequence[beta_index+1][..]]);
                            let tmp = &betas.row(beta_index+1)*&bmul;
                            let new_values = tmp.dot(&a_t);
                            betas.row_mut(beta_index).assign(&new_values);
                        }
                        // No need to normalize at the first iteration
                        let alpha_sum = alphas.row(t).sum();
                        assert!(alpha_sum != 0.0 && alpha_sum.is_finite());
                        alphas.row_mut(t).mapv_inplace(|x| x/alpha_sum);
                        self.check_row_sum(alphas.row(t));
                        let beta_sum = betas.row(beta_index).sum();
                        assert!(beta_sum != 0.0 && beta_sum.is_finite());
                        betas.row_mut(beta_index).mapv_inplace(|x| x/beta_sum);
                        self.check_row_sum(betas.row(beta_index));
                    }

                    if t >= beta_index {
                        let r = &alphas.row(t)*&betas.row(t);
                        let s = r.sum();
                        gammas.row_mut(t).assign(&(&r / s));

                        if t != beta_index {
                            let r = &alphas.row(beta_index)*&betas.row(beta_index);
                            let s = r.sum();
                            gammas.row_mut(beta_index).assign(&(&r / s));
                        }

                        for state in 0..nstates {
                            if t < sequence.len() - 1 {
                                let bmul = self.b.map(|r| r[&sequence[t+1][..]]);
                                let r = alphas[[t, state]]*&self.a.row(state)*&betas.row(t+1)*&bmul;
                                xis.slice_mut(s![t, state, ..]).assign(&(&r / s));
                            }
                            if t != beta_index && beta_index < sequence.len() - 1 {
                                let bmul = self.b.map(|r| r[&sequence[beta_index+1][..]]);
                                let r = alphas[[beta_index, state]]*&self.a.row(state)*&betas.row(beta_index+1)*&bmul;
                                xis.slice_mut(s![beta_index, state, ..]).assign(&(&r / s));
                            }
                        }
                        if t < sequence.len() - 1 {
                            let s = xis.slice(s![t, .., ..]).sum();
                            assert!(s.is_finite());
                            let m = &xis.slice(s![t, .., ..]) / s;
                            xis.slice_mut(s![t, .., ..]).assign(&m);
                        }
                        if t != beta_index && beta_index < sequence.len() - 1 {
                            let s = xis.slice(s![beta_index, .., ..]).sum();
                            assert!(s.is_finite());
                            let m = &xis.slice(s![beta_index, .., ..]) / s;
                            xis.slice_mut(s![beta_index, .., ..]).assign(&m);
                        }
                    }
                }

                pi_num += &gammas.row(1);
                omega_num += &gammas.row(sequence.len()-1);

                for state_from in 0..nstates {
                    a_den[state_from] += gammas.slice(s![..sequence.len() - 1, state_from]).sum();
                    b_den[state_from] += gammas.slice(s![..sequence.len(), state_from]).sum();
                    for state_to in 0..nstates {
                        a_num[[state_from, state_to]] += xis.slice(s![..sequence.len() - 1, state_from, state_to]).sum();
                    }
                }

                for t in 0..sequence.len() {
                    for state in 0..nstates {
                        b_num[state][&sequence[t][..]] += &gammas[[t, state]];
                    }
                }
            }


            let mut delta_a = 0.0;
            let mut delta_b = 0.0;
            let mut delta_pi = 0.0;
            for state in 0..nstates {
                let a_row = &a_num.row(state) / a_den[state];
                let a_diff = &a_row - &self.a.row(state);
                delta_a += a_diff.map(|x| x.abs()).sum();
                self.a.row_mut(state).assign(&a_row);
                assert!(self.check_row_sum(self.a.row(state)));

                let b_row = &b_num[state] / b_den[state];
                let b_diff = &b_row - &self.b[state];
                delta_b += b_diff.map(|x| x.abs()).sum();
                self.b[state] = b_row;
            }
            let pi_assign = &pi_num / sequences.len() as f64;
            let pi_diff = &pi_assign - &self.pi;
            delta_pi += pi_diff.map(|x| x.abs()).sum();
            self.pi.assign(&pi_assign);
            assert!(self.check_row_sum(self.pi.view()));

            
            let changes = delta_a + delta_b + delta_pi;
            println!("total changes: {:.2}", changes);
            let time_it = start_it.elapsed().as_secs();
            println!("Iteration in {} seconds", time_it);
            println!("<======================>");
            if changes <= tol {
                break;
            }
            a_t = self.a.t();
        }

    }

    pub fn nstates(&self) -> usize {
        self.a.nrows()
    }

    fn log(&self, p: f64) -> f64 {
        if p == 0.0 {
            f64::NEG_INFINITY
        } else {
            p.log(10.0)
        }
    }

    pub fn init_prob(&self, state: usize, obs: [usize; D]) -> f64 {
        self.log(self.pi[state]*self.b[state][&obs[..]])
    }

    pub fn transition_prob(&self, state_from: usize, state_to: usize, obs: [usize; D]) -> f64 {
        self.log(self.a[[state_from, state_to]]*self.b[state_to][&obs[..]])
    }

    pub fn transitions_to(&self, state_to: usize) -> Array1<f64> {
        let ret = self.a.slice(s![.., state_to]);
        let ret = ret.map(|x| self.log(*x));
        ret
    }

    pub fn emit_prob(&self, state: usize, obs: [usize; D]) -> f64 {
        self.log(self.b[state][&obs[..]])
    }

}
