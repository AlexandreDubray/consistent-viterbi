use gurobi::*;
use ndarray::Array1;
use std::collections::HashMap;
use rand::Rng;
use std::time::Instant;

use super::hmm::HMM;
use super::constraints::Constraints;

// (state_from, state_to, time_to)
type Arc = (usize, usize, usize);

pub struct GlobalOpti<'a> {
    hmm: &'a HMM,
    sequences: &'a Array1<Array1<usize>>,
    constraints: &'a Constraints,
    model: Model,
    vars: Array1<HashMap<Arc, Var>>,
}

impl<'b> GlobalOpti<'b> {

    pub fn new(hmm: &'b HMM, sequences: &'b Array1<Array1<usize>>, constraints: &'b Constraints) -> Self {
        let env = Env::new("logfile.log").unwrap();
        let model = Model::new("model", &env).unwrap();
        let vars = sequences.map(|_| -> HashMap<Arc, Var> { HashMap::new() });
        Self {hmm, sequences, constraints, model, vars}
    }

    fn get_in_out_flow(&self, vars: &HashMap<Arc, Var>, state: usize, time: usize) -> LinExpr {
        let mut in_out_flow = LinExpr::new();
        for other_state in 0..self.hmm.nstates() {
            let arc_in = (other_state, state, time);
            let arc_out = (state, other_state, time+1);
            match vars.get(&arc_in) {
                Some(v) => in_out_flow = in_out_flow.add_term(1.0, v.clone()),
                None => ()
            };
            match vars.get(&arc_out) {
                Some(v) => in_out_flow = in_out_flow.add_term(-1.0, v.clone()),
                None => ()
            };
        }
        in_out_flow
    }

    pub fn build_model(&mut self, prop_consistency_cstr: f64) {
        let mut objective = LinExpr::new();
        let mut inflow_map: HashMap<(usize, usize, usize), LinExpr> = HashMap::new();
        for i in 0..self.sequences.len() {
            let sequence = &self.sequences[i];
            let mut source_flow = LinExpr::new();
            let mut target_flow = LinExpr::new();
            for t in 0..sequence.len() + 1 {

                if t == 0 {
                    let p = self.hmm.init_prob(sequence[t]);
                    for state in 0..self.hmm.nstates() {
                        let proba = p[state];
                        if proba > f64::NEG_INFINITY {
                            let arc = (0, state, t);
                            let x = self.model.add_var("", Binary, proba, 0.0, 1.0, &[], &[]).unwrap();
                            source_flow = source_flow.add_term(1.0, x.clone());
                            objective = objective.add_term(proba, x.clone());
                            let mut inflow = LinExpr::new();
                            inflow = inflow.add_term(1.0, x.clone());
                            inflow_map.insert((i, state, t), inflow);
                            self.vars[i].insert(arc, x);
                        }
                    }
                } else if t == sequence.len() {
                    for state in 0..self.hmm.nstates() {
                        let arc = (state, 0, t);
                        let p = 0.0;
                        let x = self.model.add_var("", Binary, p, 0.0, 1.0, &[], &[]).unwrap();
                        target_flow = target_flow.add_term(1.0, x.clone());
                        self.vars[i].insert(arc, x);
                    }
                } else {
                    for state in 0..self.hmm.nstates() {
                        let emit_prob = self.hmm.emit_prob(state, sequence[t]);
                        if emit_prob > f64::NEG_INFINITY {
                            let transition_probs = self.hmm.transition(state);
                            let mut inflow = LinExpr::new();
                            for state_from in 0..self.hmm.nstates() {
                                let proba = transition_probs[state_from] + emit_prob;
                                if proba > f64::NEG_INFINITY {
                                    let arc = (state_from, state, t);
                                    let x = self.model.add_var("", Binary, proba, 0.0, 1.0, &[], &[]).unwrap();
                                    inflow = inflow.add_term(1.0, x.clone());
                                    objective = objective.add_term(proba, x.clone());
                                    self.vars[i].insert(arc, x);
                                }
                            }
                            inflow_map.insert((i, state, t), inflow);
                        }
                    }
                }
            }
            match self.model.add_constr("", source_flow, Equal, 1.0) {
                Ok(_) => (),
                Err(error) => panic!("Can not add constraint to the model: {:?}", error)
            };
            match self.model.add_constr("", target_flow, Equal, 1.0) {
                Ok(_) => (),
                Err(error) => panic!("Can not add constraint to the model: {:?}", error)
            };
        }


        for i in 0..self.sequences.len() {
            let sequence = &self.sequences[i];
            for t in 0..sequence.len() {
                for state in 0..self.hmm.nstates() {
                    if self.hmm.can_emit(state, sequence[t], t) {
                        let in_out_flow = self.get_in_out_flow(&self.vars[i], state, t);
                        match self.model.add_constr("", in_out_flow, Equal, 0.0) {
                            Ok(_) => (),
                            Err(error) => panic!("Cannot add constraint to the model: {:?}", error)
                        };
                    }
                }
            }
        }

        let mut rng = rand::thread_rng();
        for component in &self.constraints.components {
            for i in 0..component.len()-1 {
                let x: f64 = rng.gen();
                if x <= prop_consistency_cstr {
                    let (s1, t1) = component[i];
                    let (s2, t2) = component[i+1];
                    for state in 0..self.hmm.nstates() {
                        let mut found = false;
                        let inflow_s1 = match inflow_map.get(&(s1, state, t1)) {
                            Some(f) => {
                                found = true;
                                f.clone()
                            },
                            None => LinExpr::new()
                        };
                        let inflow_s2 = match inflow_map.get(&(s2, state, t2)) {
                            Some(f) => {
                                found = true;
                                f.clone()
                            },
                            None => LinExpr::new()
                        };
                        if found {
                            let diff_flow = inflow_s1 - inflow_s2;
                            match self.model.add_constr("", diff_flow, Equal, 0.0) {
                                Ok(_) => (),
                                Err(error) => panic!("Cannot add constraint to the model: {:?}", error)
                            };
                        }
                    }
                }
            }
        }

        match self.model.update() {
            Ok(_) => (),
            Err(error) => panic!("Can not update the model: {:?}", error)
        };
        self.model.set_objective(objective, Maximize).unwrap();
    }

    pub fn solve(&mut self) -> u64 {
        let start = Instant::now();
        match self.model.optimize() {
            Ok(_) => (),
            Err(error) => panic!("Could not solve the model: {:?}", error)
        };
        start.elapsed().as_secs()
    }

    fn get_solution(&self, seq_id: usize) -> Array1<usize> {
        let mut sol = Array1::zeros(self.sequences[seq_id].len());
        let vs = &self.vars[seq_id];
        let mut state_from = 0;
        for t in 0..sol.len() {
            for state in 0..self.hmm.nstates() {
                let arc = (state_from, state, t);
                match vs.get(&arc) {
                    Some(v) => {
                        let value = self.model.get_values(attr::X, &[v.clone()]).unwrap()[0];
                        assert!(value == 1.0 || value == 0.0);
                        if value == 1.0 {
                            sol[t] = state;
                            state_from = state;
                            break;
                        }
                    },
                    None => ()
                };
            }
        }
        sol
    }

    pub fn get_solutions(&self)  -> Array1<Array1<usize>> {
        Array1::from_iter(0..self.sequences.len()).map(|seq_id| -> Array1<usize> { self.get_solution(*seq_id) })
    }

}
