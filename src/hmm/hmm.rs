use ndarray::prelude::*;
use ndarray::{Data, DataMut};
use rand::prelude::*;
use std::time::Instant;
use std::path::PathBuf;
use serde::{Serialize, Deserialize, Deserializer};
use std::fs::{write, read_to_string};


#[derive(Serialize, Deserialize)]
pub struct HMM<const D: usize> {
    #[serde(deserialize_with="parse_transmat")]
    pub a: Array2<f64>,
    #[serde(deserialize_with="parse_bmat")]
    pub b: Array1<Array<f64, IxDyn>>,
    #[serde(deserialize_with="parse_pi")]
    pub pi: Array1<f64>,
}

impl<const D: usize> HMM<D> {

    pub fn new(nstates: usize, bdims: [usize; D]) -> Self {
        let mut rng = thread_rng();
        let mut a = Array2::from_shape_fn((nstates, nstates), |_| rng.gen::<f64>()).normalize_rows();
        let mut b = Array1::from_shape_fn(nstates, |_| Array::from_shape_fn(&bdims[..], |_| rng.gen::<f64>()).normalize());
        let mut pi = Array1::from_shape_fn(nstates, |_| rng.gen::<f64>()).normalize();
        Self { a, b, pi }
    }

    pub fn maximum_likelihood_estimation(&mut self, sequences: &Vec<Vec<[usize; D]>>, tags: &Vec<Vec<usize>>) {
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

        self.log();
    }

    fn check_row_sum(&self, row: ArrayView1<f64>) -> bool {
        let d = row.sum() - 1.0;
        d.abs() <= 0.0001
    }

    pub fn train(&mut self, sequences: &Vec<Vec<[usize; D]>>, tags: &Vec<Vec<Option<usize>>>, max_iter: usize, tol: f64) {
        let nstates = self.a.nrows();
        let r = sequences.len();
        for iter in 0..max_iter {
            let a_t = self.a.t();
            println!("[train hmm] iteration {}/{}", iter, max_iter);
            let start = Instant::now();

            let alphas: Vec<Array2<f64>> = (0..r).map(|sid| {
                let sequence = &sequences[sid];
                let tag = &tags[sid];
                let mut a = Array2::from_elem((sequence.len(), nstates), 0.0);
                match tag[0] {
                    Some(state) => a[[0, state]] = 1.0,
                    None => {
                        let bmul = self.b.map(|r| r[&sequence[0][..]]);
                        let v = (&self.pi * & bmul).normalize();
                        a.row_mut(0).assign(&v);
                    }
                };
                for t in 1..sequence.len() {
                    match tag[t] {
                        Some(state) => a[[t, state]] = 1.0,
                        None => {
                            let bmul = self.b.map(|r| r[&sequence[t][..]]);
                            let new_values = (&a.row(t-1) * &bmul).dot(&self.a).normalize();
                            a.row_mut(t).assign(&new_values);
                        }
                    };
                }
                a
            }).collect();

            let betas: Vec<Array2<f64>> = (0..r).map(|sid| {
                let sequence = &sequences[sid];
                let tag = &tags[sid];
                let mut b = Array2::from_elem((sequence.len(), nstates), 0.0);
                match tag[sequence.len() - 1] {
                    Some(state) => b[[sequence.len() - 1, state]] = 1.0,
                    None => b.row_mut(sequence.len() - 1).fill(1.0)
                };
                for t in (0..sequence.len()-1).rev() {
                    match tag[t] {
                        Some(state) => b[[t, state]] = 1.0,
                        None => {
                            let bmul = self.b.map(|r| r[&sequence[t+1][..]]);
                            let tmp = &b.row(t+1) * &bmul;
                            let new_values = tmp.dot(&a_t).normalize();
                            b.row_mut(t).assign(&new_values);
                        }
                    };
                }
                b
            }).collect();

            let gammas: Vec<Array2<f64>> = (0..r).map(|sid| {
                let n = sequences[sid].len();
                let mut g = Array2::from_elem((n, nstates), 0.0);
                for t in 0..n {
                    let p = (&alphas[sid].row(t) * &betas[sid].row(t)).normalize();
                    g.row_mut(t).assign(&p);
                }
                g
            }).collect();

            let xis: Vec<Vec<Array2<f64>>> = (0..r).map(|sid| {
                let sequence = &sequences[sid];
                (0..sequence.len()-1).map(|t| {
                    let alpha_m = Array2::from_shape_fn((nstates, nstates), |x| alphas[sid][[t, x.0]]);
                    let bmul = self.b.map(|r| r[&sequence[t+1][..]]);
                    let m = &(&bmul * &betas[sid].row(t+1)) * &alpha_m;
                    (&m * &self.a).normalize()
                }).collect()
            }).collect();

            self.pi = gammas.iter().fold(Array1::from_elem(nstates, 0.0), |sum, val| sum + val.row(0)) / r as f64;

            let a_den = gammas.iter().fold(Array1::from_elem(nstates, 0.0), |sum, val| {
                sum + Array1::from_shape_fn(nstates, |i| val.slice(s![..-1, i]).sum())
            });
            let a_div = Array2::from_shape_fn((nstates, nstates), |x| a_den[x.0]);


            self.a = xis.iter().fold(Array2::from_elem((nstates, nstates), 0.0), |sum, val| {
                sum + val.iter().fold(Array2::from_elem((nstates, nstates), 0.0), |sum2, val2| sum2 + val2)
            });
            self.a /= &a_div;

            for state in 0..nstates {
                self.b[state].fill(0.0);
            }
            for sid in 0..sequences.len() {
                let sequence = &sequences[sid];
                for t in 0..sequence.len() {
                    let obs = &sequence[t];
                    for state in 0..nstates {
                        self.b[state][&obs[..]] += gammas[sid][[t, state]];
                    }
                }
            }
            let b_den = gammas.iter().fold(Array1::from_elem(nstates, 0.0), |sum, val| {
                let t = val.nrows();
                sum + Array1::from_shape_fn(nstates, |i| val.slice(s![.., i]).sum())
            });
            for state in 0..nstates {
                self.b[state] /= b_den[state];
            }
            let elapsed = start.elapsed().as_secs();
            println!("[train hmm]\t {} seconds", elapsed);
        }
        self.log();
    }

    fn log(&mut self) {
        let map_log = |x: f64| -> f64 {
            if x == 0.0 {
                f64::NEG_INFINITY
            } else {
                x.log(10.0)
            }
        };
        self.a.mapv_inplace(map_log);
        for state in 0..self.b.len() {
            self.b[state].mapv_inplace(map_log);
        }
        self.pi.mapv_inplace(map_log);
    }

    pub fn nstates(&self) -> usize {
        self.a.nrows()
    }

    pub fn init_prob(&self, state: usize, obs: [usize; D]) -> f64 {
        self.pi[state] + self.b[state][&obs[..]]
    }

    pub fn transition_prob(&self, state_from: usize, state_to: usize, obs: [usize; D]) -> f64 {
        self.a[[state_from, state_to]] + self.b[state_to][&obs[..]]
    }

    pub fn transitions_to(&self, state_to: usize) -> ArrayView1<f64> {
        self.a.column(state_to)
    }

    pub fn emit_prob(&self, state: usize, obs: [usize; D]) -> f64 {
        self.b[state][&obs[..]]
    }

    pub fn write(&self, opath: &mut PathBuf) {
        opath.set_file_name("hmm.json");
        let serialized = serde_json::to_string(&self).unwrap();
        write(opath, serialized).unwrap();
    }

    pub fn from_json(ipath: &mut PathBuf) -> Self {
        ipath.set_file_name("hmm.json");
        let serialized = read_to_string(ipath).unwrap();
        serde_json::from_str(&serialized).unwrap()
    }
}

fn parse_transmat<'de, A>(d: A) -> Result<Array2<f64>, A::Error> where A: Deserializer<'de> {
    Deserialize::deserialize(d).map(|x: Array2<Option<_>>| {
        x.map(|y: &Option<_>| {
            y.unwrap_or(f64::NEG_INFINITY)
        })
    })
}

fn parse_bmat<'de, A>(d: A) -> Result<Array1<Array<f64, IxDyn>>, A::Error> where A: Deserializer<'de> {
    Deserialize::deserialize(d).map(|x: Array1<Array<Option<f64>, IxDyn>>| {
        x.map(|y| y.map(|v| v.unwrap_or(f64::NEG_INFINITY)))
    })
}

fn parse_pi<'de, A>(d: A) -> Result<Array1<f64>, A::Error> where A: Deserializer<'de> {
    Deserialize::deserialize(d).map(|x: Array1<Option<f64>>| x.map(|v| v.unwrap_or(f64::NEG_INFINITY)))
}

trait Array1Norm {
    fn normalize(self) -> Self;
}

impl<S> Array1Norm for ArrayBase<S, Ix1>
where
    S: DataMut + Data<Elem = f64>,
{
    fn normalize(mut self) -> Self {
        let s = self.sum();
        if s !=  0.0 {
            self /= s;
        } else {
            self.fill(1.0 / self.len() as f64);
        }
        self
    }
}

trait Array2Norm {
    fn normalize_rows(self) -> Self;
    fn normalize(self) -> Self;
}

impl<S> Array2Norm for ArrayBase<S, Ix2>
where
    S: DataMut + Data<Elem = f64>,
{
    fn normalize_rows(mut self) -> Self {
        for mut row in self.rows_mut() {
            let s = row.sum();
            if s != 0.0 {
                row /= s;
            } else {
                row.fill(1.0 / row.len() as f64);
            }
        }
        self
    }

    fn normalize(mut self) -> Self {
        let s = self.sum();
        if s != 0.0 {
            self /= s;
        } else {
            let d = (self.nrows() *  self.ncols()) as f64;
            self.fill(1.0 / d);
        }
        self
    }
}

trait ArrayDynNorm {
    fn normalize(self) -> Self;
}

impl<S> ArrayDynNorm for ArrayBase<S, IxDyn>
where
    S: DataMut + Data<Elem = f64>,
{
    fn normalize(mut self) -> Self {
        let s = self.sum();
        if s != 0.0 {
            self /= s;
        } else {
            let nb_elem = self.raw_dim().as_array_view().fold(1, |p, val| p*val) as f64;
            self.fill(1.0 / nb_elem);
        }
        self
    }
}
