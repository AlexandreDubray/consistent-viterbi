use gurobi::*;
use ndarray::Array1;
use std::collections::HashMap;
use rand::Rng;
use std::time::Instant;

use super::hmm::HMM;
use super::constraints::Constraints;

pub struct GlobalOpti<'a> {
    hmm: &'a HMM,
    sequences: &'a Array1<Array1<usize>>,
    constraints: &'a Constraints,
    model: Model,
    consistency_constraints: Vec<Constr>,
    inflow_map: HashMap<(usize, usize, usize), LinExpr>,
    solutions: Array1<Array1<usize>>
}

impl<'b> GlobalOpti<'b> {

    pub fn new(hmm: &'b HMM, sequences: &'b Array1<Array1<usize>>, constraints: &'b Constraints) -> Self {
        let mut env = Env::new("logfile.log").unwrap();
        env.set(param::OutputFlag, 0).unwrap();
        let model = Model::new("model", &env).unwrap();
        let consistency_constraints: Vec<Constr> = Vec::new();
        let inflow_map: HashMap<(usize, usize, usize), LinExpr> = HashMap::new();
        let solutions = Array1::from_iter(0..sequences.len()).map(|seq_id| -> Array1<usize> { Array1::from_elem(sequences[*seq_id].len(), 0) });
        Self {hmm, sequences, constraints, model, consistency_constraints, inflow_map, solutions}
    }

    fn add_var(&mut self, p: f64, name: String) -> Var {
        self.model.add_var(&name, Binary, p, 0.0, 1.0, &[], &[]).unwrap()
    }

    pub fn build_model(&mut self) {
        let mut objective = LinExpr::new();
        for i in 0..self.sequences.len() {
            let sequence = &self.sequences[i];
            let mut source_flow = LinExpr::new();
            let mut target_flow = LinExpr::new();

            let mut next_inflows: Array1<LinExpr> = Array1::from_elem(self.hmm.nstates(), LinExpr::new());
            let mut current_inflows: Array1<LinExpr> = Array1::from_elem(self.hmm.nstates(), LinExpr::new());
            for t in 0..sequence.len() {
                for state in 0..self.hmm.nstates() {
                    let emit_prob = self.hmm.emit_prob(state, sequence[t]);
                    if emit_prob > f64::NEG_INFINITY {
                        let mut inflow = current_inflows[state].clone();
                        let mut outflow = LinExpr::new();
                        if t == 0 {
                            // Variable from source to state node
                            let from_source_p = self.hmm.init_prob(state, sequence[0]);
                            if from_source_p > f64::NEG_INFINITY {
                                let name = format!("{}-{}-{}-{}", i, 0, state, t);
                                let from_source = self.add_var(from_source_p, name);
                                source_flow += from_source.clone();
                                inflow += from_source.clone();
                                objective += from_source.clone()*from_source_p;

                                if sequence.len() == 1 {
                                    // Also the last layer
                                    let name = format!("{}-{}-{}-{}", i, state, 0, t+1);
                                    let arc_out = self.add_var(0.0, name);
                                    outflow += arc_out.clone();
                                    target_flow += arc_out.clone();
                                } else {
                                    // Variable to states of the next layer
                                    let transitions = self.hmm.transitions_from(state);
                                    for state_to in 0..self.hmm.nstates() {
                                        let arc_p = transitions[state_to] + self.hmm.emit_prob(state_to, sequence[t+1]);
                                        if arc_p > f64::NEG_INFINITY {
                                            let name = format!("{}-{}-{}-{}", i, state, state_to, t+1);
                                            let arc_var = self.add_var(arc_p, name);
                                            outflow += arc_var.clone();
                                            next_inflows[state_to] += arc_var.clone();
                                            objective += arc_var.clone()*arc_p;
                                        }
                                    }
                                }
                            }
                        } else if t == sequence.len() - 1 {
                            let name = format!("{}-{}-{}-{}", i, state, 0, t+1);
                            let arc_out = self.add_var(0.0, name);
                            outflow += arc_out.clone();
                            target_flow += arc_out.clone();
                        } else {
                            let transitions = self.hmm.transitions_from(state);
                            for state_to in 0..self.hmm.nstates() {
                                let arc_p = transitions[state_to] + self.hmm.emit_prob(state_to, sequence[t+1]);
                                if arc_p > f64::NEG_INFINITY {
                                    let name = format!("{}-{}-{}-{}", i, state, state_to, t+1);
                                    let arc_var = self.add_var(arc_p, name);
                                    outflow += arc_var.clone();
                                    next_inflows[state_to] += arc_var.clone();
                                    objective += arc_var.clone()*arc_p;
                                }
                            }
                        }
                        if self.constraints.constrained_elements.contains(&(i, t)) {
                            self.inflow_map.insert((i, t, state), inflow.clone());
                        }
                        let diff_flow = inflow - outflow;
                        self.model.add_constr("", diff_flow, Equal, 0.0).unwrap();
                    }
                }
                current_inflows = next_inflows;
                next_inflows = Array1::from_elem(self.hmm.nstates(), LinExpr::new());
            }
            self.model.add_constr("", source_flow, Equal, 1.0).unwrap();
            self.model.add_constr("", target_flow, Equal, 1.0).unwrap();
        }

        match self.model.update() {
            Ok(_) => (),
            Err(error) => panic!("Can not update the model: {:?}", error)
        };
        self.model.set_objective(objective, Maximize).unwrap();
    }

    pub fn solve(&mut self, prop_consistency_cstr: f64) -> u64 {
        self.model.reset().unwrap();
        for cstr in &mut self.consistency_constraints {
            cstr.remove();
        }
        self.consistency_constraints.clear();
        let mut rng = rand::thread_rng();
        for component in &self.constraints.components {
            for i in 0..component.len()-1 {
                let x: f64 = rng.gen();
                if x <= prop_consistency_cstr {
                    let (s1, t1) = component[i];
                    let (s2, t2) = component[i+1];
                    for state in 0..self.hmm.nstates() {
                        let mut found = false;
                        let inflow_s1 = match self.inflow_map.get(&(s1, state, t1)) {
                            Some(f) => {
                                found = true;
                                f.clone()
                            },
                            None => LinExpr::new()
                        };
                        let inflow_s2 = match self.inflow_map.get(&(s2, state, t2)) {
                            Some(f) => {
                                found = true;
                                f.clone()
                            },
                            None => LinExpr::new()
                        };
                        if found {
                            let diff_flow = inflow_s1 - inflow_s2;
                            let cstr = self.model.add_constr("", diff_flow, Equal, 0.0).unwrap();
                            self.consistency_constraints.push(cstr);
                        }
                    }
                }
            }
        }
        self.model.update().unwrap();
        let start = Instant::now();
        match self.model.optimize() {
            Ok(_) => (),
            Err(error) => panic!("Could not solve the model: {:?}", error)
        };
        start.elapsed().as_secs()
    }

    fn arc_from_name(&self, var: &Var) -> (usize, usize, usize) {
        let name = &self.model.get_values(attr::VarName, &[var.clone()]).unwrap()[0];
        let mut split = name.split("-");
        let seq_id: usize = split.next().unwrap().parse().unwrap();
        let _state_from: usize = split.next().unwrap().parse().unwrap();
        let state_to: usize = split.next().unwrap().parse().unwrap();
        let t: usize = split.next().unwrap().parse().unwrap();
        (seq_id, state_to, t)
    }

    pub fn get_solutions(&mut self) -> &Array1<Array1<usize>> {
        for var in self.model.get_vars() {
            let value = self.model.get_values(attr::X, &[var.clone()]).unwrap()[0];
            if value == 1.0 {
                let (seq_id, state_to, t) = self.arc_from_name(var);
                if t < self.sequences[seq_id].len() {
                    self.solutions[seq_id][t] = state_to;
                }
            }

        }
        &self.solutions
    }
}
