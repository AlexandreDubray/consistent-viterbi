use gurobi::*;
use ndarray::Array1;
use std::time::Instant;

use super::super::hmm::hmm::HMM;
use super::utils::SuperSequence;

pub struct GlobalOpti<'a, const D: usize> {
    hmm: &'a HMM<D>,
    sequence: &'a SuperSequence<'a, D>,
    model: Model,
    solution: Array1<usize>,
}

impl<'b, const D: usize> GlobalOpti<'b, D> {

    pub fn new(hmm: &'b HMM<D>, sequence: &'b SuperSequence<D>) -> Self {
        let mut env = Env::new("logfile.log").unwrap();
        //env.set(param::OutputFlag, 0).unwrap();
        let model = Model::new("model", &env).unwrap();
        let solution = Array1::from_elem(sequence.len(), 0);
        Self {hmm, sequence, model, solution}
    }

    fn add_var(&mut self, p: f64, idx: usize, state_from: usize, state_to: usize) -> Var {
        let name = format!("{}-{}-{}", idx, state_from, state_to);
        self.model.add_var(&name, Binary, p, 0.0, 1.0, &[], &[]).unwrap()
    }

    fn compute_outflow(&mut self, state_from: usize, idx: i32, inflow_cache: &mut Vec<Option<LinExpr>>, layer_edges: &mut Vec<Var>) -> LinExpr {
        let mut outflow = LinExpr::new();
        let n_idx = (idx + 1) as usize;
        for other_state in 0..self.hmm.nstates() {
            let next_elem = &self.sequence[n_idx];
            let arc_p = next_elem.arc_p(self.hmm, state_from, other_state);
            if arc_p > f64::NEG_INFINITY {
                let arc = self.add_var(arc_p, n_idx, state_from, other_state);
                layer_edges.push(arc.clone());
                outflow += arc.clone();
                match &mut inflow_cache[other_state] {
                    Some(f) => {
                        *f += arc.clone();
                    },
                    None => {
                        let mut f = LinExpr::new();
                        f += arc.clone();
                        inflow_cache[other_state] = Some(f);
                    }
                };
            }
        }
        outflow
    }

    fn get_sum_layer(&self, v: &mut Vec<Var>) -> LinExpr {
        let mut s = LinExpr::new();
        while !v.is_empty() {
            let edge = v.pop().unwrap();
            s += edge;
        }
        s
    }

    pub fn build_model(&mut self) {

        let mut cstr_inflow_cache: Vec<Vec<Option<LinExpr>>> = (0..self.sequence.nb_cstr).map(|_| -> Vec<Option<LinExpr>> {
            (0..self.hmm.nstates()).map(|_| None).collect()
        }).collect();
        let mut inflow_cache: Vec<Option<LinExpr>> = (0..self.hmm.nstates()).map(|_| None).collect();
        let mut inflow_tmp: Vec<Option<LinExpr>> = (0..self.hmm.nstates()).map(|_| None).collect();

        let mut edge_layer: Vec<Var> = Vec::new();
        let source_flow = self.compute_outflow(0, -1, &mut inflow_cache, &mut edge_layer);
        self.model.add_constr("", source_flow, Equal, 1.0).unwrap();
        let c = self.get_sum_layer(&mut edge_layer);
        self.model.add_constr("", c, Equal, 1.0).unwrap();
        let mut target_flow = LinExpr::new();


        for idx in 0..self.sequence.len() {
            if idx % 1000 == 0 {
                println!("{}/{}", idx+1, self.sequence.len());
            }
            let element = &self.sequence[idx];
            let is_constrained = element.constraint_component != -1;
            if idx != self.sequence.len() - 1 {
                for i in (0..self.hmm.nstates()).rev() {
                    inflow_tmp[i] = inflow_cache.remove(i);
                }

                for _ in 0..self.hmm.nstates() {
                    inflow_cache.push(None);
                }

                for state in 0..self.hmm.nstates() {
                    match inflow_tmp[state].as_ref() {
                        Some(inflow) => {
                            if is_constrained {
                                let ucomp = element.constraint_component as usize;
                                match &cstr_inflow_cache[ucomp][state] {
                                    Some(f) => {
                                        let dflow = (*f).clone() - (*inflow).clone();
                                        self.model.add_constr("", dflow, Equal, 0.0).unwrap();
                                        cstr_inflow_cache[ucomp][state] = Some(inflow.clone());
                                    },
                                    None => cstr_inflow_cache[ucomp][state] = Some(inflow.clone())
                                };
                            }
                            let outflow = self.compute_outflow(state, idx as i32, &mut inflow_cache, &mut edge_layer);
                            let diff_flow = (*inflow).clone() - outflow;
                            self.model.add_constr("", diff_flow, Equal, 0.0).unwrap();
                        },
                        None => ()
                    }
                }
                let c = self.get_sum_layer(&mut edge_layer);
                self.model.add_constr("", c, Equal, 1.0).unwrap();
            } else {
                for state in 0..self.hmm.nstates() {
                    match &inflow_cache[state] {
                        Some(inflow) => {
                            if is_constrained {
                                match &cstr_inflow_cache[element.constraint_component as usize][state] {
                                    Some(inflow_eq) => {
                                        let dflow = (*inflow_eq).clone() - (*inflow).clone();
                                        self.model.add_constr("", dflow, Equal, 0.0).unwrap();
                                    },
                                    None => ()
                                };
                            }
                            let arc = self.add_var(0.0, idx+1, state, 0);
                            target_flow += arc.clone();
                            let mut outflow = LinExpr::new();
                            outflow += arc.clone();
                            let diff_flow = (*inflow).clone() - outflow;
                            self.model.add_constr("", diff_flow, Equal, 0.0).unwrap();
                        },
                        None => ()
                    };
                }
            }
        }
        self.model.add_constr("", target_flow, Equal, 1.0).unwrap();
        self.model.update().unwrap();
        let mut objective = LinExpr::new();
        let vars: Vec<Var> = self.model.get_vars().map(|x| (*x).clone()).collect();
        let coefs: Vec<f64> = (0..vars.len()).map(|i| vars[i].get(&self.model, attr::Obj).unwrap()).collect();
        objective = objective.add_terms(&coefs, &vars);
        self.model.set_objective(objective, Maximize).unwrap();
    }

    pub fn solve(&mut self) -> u128 {
        let start = Instant::now();
        self.model.optimize().unwrap();
        start.elapsed().as_millis()
    }

    fn arc_from_name(&self, var: &Var) -> (usize, usize) {
        let name = &self.model.get_values(attr::VarName, &[var.clone()]).unwrap()[0];
        let mut split = name.split("-");
        let idx: usize = split.next().unwrap().parse().unwrap();
        let _state_from: usize = split.next().unwrap().parse().unwrap();
        let state_to: usize = split.next().unwrap().parse().unwrap();
        (idx, state_to)
    }

    pub fn get_solutions(&mut self) -> &Array1<usize> {
        for var in self.model.get_vars() {
            let value = self.model.get_values(attr::X, &[var.clone()]).unwrap()[0];
            if value == 1.0 {
                let (idx, state_to) = self.arc_from_name(var);
                if idx < self.sequence.len() {
                    self.solution[idx] = state_to;
                }
            }

        }
        &self.solution
    }
}
