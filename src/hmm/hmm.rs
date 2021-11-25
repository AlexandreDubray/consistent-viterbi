use ndarray::{Array2, Array1, s, Array3, ArrayView1};
use ndarray_stats::QuantileExt;
use rand::prelude::*;
use std::time::Instant;

pub struct HMM {
    pub a: Array2<f64>,
    pub b: Array2<f64>,
    pub pi: Array1<f64>,
    omega: Array1<f64>,
}

impl HMM {

    pub fn new(a: Array2<f64>, b: Array2<f64>, pi: Array1<f64>, omega: Array1<f64>) -> Self {
        // TODO: checks
        Self {a, b, pi, omega}
    }

    pub fn from_supervised(sequences: &Array1<Array1<usize>>, tags: &Array1<Array1<usize>>, nstates: usize, nobs: usize) -> Self {
        let mut a = Array2::from_elem((nstates, nstates), 0.0);
        let mut b = Array2::from_elem((nstates, nobs), 0.0);
        let mut pi = Array1::from_elem(nstates, 0.0);

        let mut seen_states = Array1::from_elem(nstates, 0.0);
        let mut omega = Array1::from_elem(nstates, 0.0);

        for i in 0..sequences.len() {
            let sequence = &sequences[i];
            let tag = &tags[i];

            pi[tag[0]] += 1.0;
            for t in 0..sequence.len() - 1 {
                b[[tag[t], sequence[t]]] += 1.0;
                a[[tag[t], tag[t+1]]] += 1.0;
                seen_states[tag[t]] += 1.0;
            }
            b[[tag[sequence.len()-1], sequence[sequence.len()-1]]] += 1.0;
            seen_states[tag[sequence.len()-1]] += 1.0;
            omega[tag[sequence.len()-1]] += 1.0;
        }

        for state in 0..nstates {
            let mut a_row = a.row_mut(state);
            if seen_states[state] != omega[state] {
                a_row /= seen_states[state] - omega[state];
            } else {
                a_row.fill(0.0); 
            }
            pi[state] = pi[state] / sequences.len() as f64;
            omega[state] = omega[state] / sequences.len() as f64;
            let mut b_row = b.row_mut(state);
            b_row /= seen_states[state];
        }

        HMM::new(a, b, pi, omega)
    }

    fn check_row_sum(row: ArrayView1<f64>) -> bool {
        let d = row.sum() - 1.0;
        d.abs() <= 0.0001
    }

    pub fn from_semi_supervised(sequences: &Array1<Array1<usize>>, tags: &Array1<Array1<Option<usize>>>, nstates: usize, nobs: usize, max_iter: usize, tol: f64) -> Self {
        let mut rng = thread_rng();
        let mut a = Array2::from_shape_fn((nstates, nstates), |_| rng.gen::<f64>());
        let mut b = Array2::from_shape_fn((nstates, nobs), |_| rng.gen::<f64>());
        let mut pi = Array1::from_shape_fn(nstates, |_| rng.gen::<f64>());
        let mut omega = Array1::from_shape_fn(nstates, |_| rng.gen::<f64>());

        for state in 0..nstates {
            let s = a.row(state).sum();
            let r = &a.row(state) / s;
            a.row_mut(state).assign(&r);
            let s = b.row(state).sum();
            let r = &b.row(state) / s;
            b.row_mut(state).assign(&r);
        }
        let mut a_t = a.t();

        let s = pi.sum();
        pi /= s;
        let s = omega.sum();
        omega /= s;
        

        let mut pi_num = Array1::from_elem(nstates, 0.0);
        let mut omega_num = Array1::from_elem(nstates, 0.0);
        let mut a_num = Array2::from_elem((nstates, nstates), 0.0);
        let mut a_den = Array1::from_elem(nstates, 0.0);
        let mut b_den = Array1::from_elem(nstates, 0.0);
        let mut b_num = Array2::from_elem((nstates, nobs), 0.0);

        let max_seq_size = *sequences.map(|x| x.len()).max().unwrap();

        let mut alphas = Array2::from_elem((max_seq_size, nstates), 0.0);
        let mut betas = Array2::from_elem((max_seq_size, nstates), 0.0);
        let mut gammas = Array2::from_elem((max_seq_size, nstates), 0.0);
        let mut xis = Array3::from_elem((max_seq_size, nstates, nstates), 0.0);

        for seq_id in 0..sequences.len() {
            for t in 0..sequences[seq_id].len() {
                match tags[seq_id][t] {
                    Some(state) => {
                        alphas[[t, state]] = 1.0;
                        betas[[t, state]] = 1.0;
                        gammas[[t, state]] = 1.0;
                    },
                    None => ()
                };
            }
        }

        for iter in 0..max_iter {
            println!("Iteration {}/{}", iter+1, max_iter);
            let start_it = Instant::now();
            pi_num.fill(0.0);
            omega_num.fill(0.0);
            a_num.fill(0.0);
            a_den.fill(0.0);
            b_den.fill(0.0);
            b_num.fill(0.0);

            for seq_id in 0..sequences.len() {
                let sequence = &sequences[seq_id];
                for t in 0..sequence.len() {
                    let beta_index = sequence.len() - 1 - t;
                    let update_t = tags[seq_id][t].is_none();
                    let update_beta_t = tags[seq_id][beta_index].is_none();
                    if !update_t && !update_beta_t {
                        continue;
                    }
                    if t == 0 {
                        if update_t {
                            let new_values = &pi*&b.column(sequence[t]);
                            alphas.row_mut(t).assign(&new_values);
                        }
                        if update_beta_t {
                            betas.row_mut(beta_index).fill(1.0);
                        }
                    } else {
                        if update_t {
                            let new_values = alphas.row(t-1).dot(&a)*&b.column(sequence[t]);
                            alphas.row_mut(t).assign(&new_values);
                        }
                        if update_beta_t {
                            let tmp = &betas.row(beta_index+1)*&b.column(sequence[beta_index+1]);
                            let new_values = tmp.dot(&a_t);
                            betas.row_mut(beta_index).assign(&new_values);
                        }
                        // No need to normalize at the first iteration
                        let alpha_sum = alphas.row(t).sum();
                        assert!(alpha_sum != 0.0 && alpha_sum.is_finite());
                        alphas.row_mut(t).mapv_inplace(|x| x/alpha_sum);
                        HMM::check_row_sum(alphas.row(t));
                        let beta_sum = betas.row(beta_index).sum();
                        assert!(beta_sum != 0.0 && beta_sum.is_finite());
                        betas.row_mut(beta_index).mapv_inplace(|x| x/beta_sum);
                        HMM::check_row_sum(betas.row(beta_index));
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
                                let r = alphas[[t, state]]*&a.row(state)*&betas.row(t+1)*&b.column(sequence[t+1]);
                                xis.slice_mut(s![t, state, ..]).assign(&(&r / s));
                            }
                            if t != beta_index && beta_index < sequence.len() - 1 {
                                let r = alphas[[beta_index, state]]*&a.row(state)*&betas.row(beta_index+1)*&b.column(sequence[beta_index+1]);
                                xis.slice_mut(s![beta_index, state, ..]).assign(&(&r / s));
                            }
                        }
                        if t < sequence.len() - 1 {
                            let s = xis.slice(s![t, .., ..]).sum();
                            assert!(s != 0.0 && s.is_finite());
                            let m = &xis.slice(s![t, .., ..]) / s;
                            xis.slice_mut(s![t, .., ..]).assign(&m);
                        }
                        if t != beta_index && beta_index < sequence.len() - 1 {
                            let s = xis.slice(s![beta_index, .., ..]).sum();
                            assert!(s != 0.0 && s.is_finite());
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
                    let slice = s![.., sequence[t]];
                    let b_assign = &b_num.slice(slice) + &gammas.row(t);
                    b_num.slice_mut(slice).assign(&b_assign);
                }
            }


            let mut delta_a = 0.0;
            let mut delta_b = 0.0;
            let mut delta_pi = 0.0;
            let mut delta_omega = 0.0;
            for state in 0..nstates {
                let a_row = &a_num.row(state) / a_den[state];
                let a_diff = &a_row - &a.row(state);
                delta_a += a_diff.map(|x| x.abs()).sum();
                a.row_mut(state).assign(&a_row);
                assert!(HMM::check_row_sum(a.row(state)));

                let b_row = &b_num.row(state) / b_den[state];
                let b_row = b_row.map(|x| if x.is_nan() { 0.0 } else { *x });
                let b_diff = &b_row - &b.row(state);
                delta_b += b_diff.map(|x| x.abs()).sum();
                b.row_mut(state).assign(&b_row);
                assert!(HMM::check_row_sum(b.row(state)));
            }
            let pi_assign = &pi_num / sequences.len() as f64;
            let pi_diff = &pi_assign - &pi;
            delta_pi += pi_diff.map(|x| x.abs()).sum();
            pi.assign(&pi_assign);
            assert!(HMM::check_row_sum(pi.view()));

            let omega_assign = &omega_num / sequences.len() as f64;
            let omega_diff = &omega_assign - &omega;
            delta_omega += omega_diff.map(|x| x.abs()).sum();
            omega.assign(&omega_assign);
            assert!(HMM::check_row_sum(omega.view()));

            
            let changes = delta_a + delta_b + delta_pi + delta_omega;
            println!("Deltas: {:.2} {:.2} {:.2} {:.2}", delta_a, delta_b, delta_pi, delta_omega);
            println!("total changes: {:.2}", changes);

            let time_it = start_it.elapsed().as_secs();
            println!("Iteration in {} seconds", time_it);
            println!("<======================>");
            if changes <= tol {
                break;
            }
            a_t = a.t();
        }

        Self {a, b, pi, omega}
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

    pub fn init_prob(&self, state: usize, obs: usize) -> f64 {
        self.log(self.pi[state]*self.b[[state, obs]])
    }

    pub fn transition_prob(&self, state_from: usize, state_to: usize, obs: usize) -> f64 {
        self.log(self.a[[state_from, state_to]]*self.b[[state_to, obs]])
    }

    pub fn transitions_to(&self, state_to: usize) -> Array1<f64> {
        let ret = self.a.slice(s![.., state_to]);
        let ret = ret.map(|x| self.log(*x));
        ret
    }

    pub fn emit_prob(&self, state: usize, obs: usize) -> f64 {
        self.log(self.b[[state, obs]])
    }

    pub fn generate_single(&self) -> (Array1<usize>, Array1<usize>) {
        let mut rng = rand::thread_rng();
        let states: Vec<usize> = (0..self.nstates()).collect();
        let observations: Vec<usize> = (0..self.b.ncols()).collect();

        let mut sequence: Vec<usize> = Vec::new();
        let mut tags: Vec<usize> = Vec::new();
        let mut current_state = states.choose_weighted(&mut rng, |state| self.pi[*state]).unwrap();
        let mut obs = observations.choose_weighted(&mut rng, |obs| self.b[[*current_state, *obs]]).unwrap();
        sequence.push(*obs);
        tags.push(*current_state);

        let mut p_end: f64 = rng.gen();
        p_end = p_end.log(10.0);
        let mut finished = p_end <= self.omega[*current_state];

        while !finished {
            current_state = states.choose_weighted(&mut rng, |state| self.a[[*current_state, *state]]).unwrap();
            obs = observations.choose_weighted(&mut rng, |obs| self.b[[*current_state, *obs]]).unwrap();
            sequence.push(*obs);
            tags.push(*current_state);
            p_end = rng.gen();
            p_end = p_end.log(10.0);
            finished = p_end <= self.omega[*current_state];
        }

        (Array1::from_vec(sequence), Array1::from_vec(tags))
    }


    pub fn generate(&self, n: usize) -> (Array1<Array1<usize>>, Array1<Array1<usize>>) {
        let mut sequences: Vec<Array1<usize>> = Vec::new();
        let mut tags: Vec<Array1<usize>> = Vec::new();
        for i in 0..n {
            if i % 100 == 0 {
                println!("{}/{}", i, n);
            }
            let (seq, hstates) = self.generate_single();
            sequences.push(seq);
            tags.push(hstates);
        }
        (Array1::from_vec(sequences), Array1::from_vec(tags))
    }

}
