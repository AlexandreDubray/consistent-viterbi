use gurobi::*;
use ndarray::Array1;
use std::collections::HashMap;

use super::hmm::HMM;
use super::constraints::Constraints;

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Arc {
    state_from: usize,
    state_to: usize,
    time_to: usize
}

pub struct GlobalOpti<'a> {
    hmm: &'a HMM,
    sequences: &'a Array1<Array1<usize>>,
    constraints: &'a Constraints,
    model: Model,
    vars: Array1<HashMap<Arc, Var>>
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
            let arc_in = Arc { state_from: other_state, state_to: state, time_to: time };
            let arc_out = Arc {state_from: state, state_to: other_state, time_to: time+1};
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

    pub fn build_model(&mut self) {

        println!("Creating the global optimisation problem");


        println!("Adding the variables in the model");
        let mut inflow_map: HashMap<(usize, usize, usize), LinExpr> = HashMap::new();
        for i in 0..self.sequences.len() {
            if i % 10 == 0 {
                println!("{}/{}", i, self.sequences.len());
            }
            let sequence = &self.sequences[i];
            let mut source_flow = LinExpr::new();
            let mut target_flow = LinExpr::new();
            for t in 0..sequence.len() + 1 {

                if t == 0 {
                    let p = self.hmm.init_prob(sequence[t]);
                    for state in 0..self.hmm.nstates() {
                        let proba = p[state];
                        if proba > f64::NEG_INFINITY {
                            let arc = Arc { state_from: 0, state_to: state, time_to: t };
                            let x = self.model.add_var("", Binary, proba, 0.0, 1.0, &[], &[]).unwrap();
                            source_flow = source_flow.add_term(1.0, x.clone());
                            let mut inflow = LinExpr::new();
                            inflow = inflow.add_term(1.0, x.clone());
                            if i == 64 && state == 312 {
                                println!("DEBUGGG");
                            }
                            inflow_map.insert((i, state, t), inflow);
                            self.vars[i].insert(arc, x);
                        }
                    }
                } else if t == sequence.len() {
                    for state in 0..self.hmm.nstates() {
                        let arc = Arc {state_from: state, state_to: 0, time_to: t};
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
                                    let arc = Arc {state_from, state_to: state, time_to:t};
                                    let x = self.model.add_var("", Binary, proba, 0.0, 1.0, &[], &[]).unwrap();
                                    inflow = inflow.add_term(1.0, x.clone());
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

        println!("Adding the shortest path constraints");

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
            if i % 10 == 0 {
                println!("{}/{}", i, self.sequences.len());
            }
        }

        println!("Adding the consistency constraints");
        for component in &self.constraints.components {
            for i in 0..component.len() {
                let (s1, t1) = component[i];
                for j in i+1..component.len() {
                    let (s2, t2) = component[j];
                    for state in 0..self.hmm.nstates() {
                        if self.hmm.can_emit(state, self.sequences[s1][t1], t1) && self.hmm.can_emit(state, self.sequences[s2][t2], t2) {
                            //println!("{} {} {} {}", s2, state, t2, self.hmm.emit_prob(state, self.sequences[s2][t2]));
                            let inflow_s1 = inflow_map.get(&(s1, state, t1)).unwrap().clone();
                            let inflow_s2 = inflow_map.get(&(s2, state, t2)).unwrap().clone();
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

        println!("updating model");
        match self.model.update() {
            Ok(_) => (),
            Err(error) => panic!("Can not update the model: {:?}", error)
        };
        println!("Writing model");
        self.model.write("global_opti.lp").unwrap();
    }

    pub fn solve(&mut self) {
        match self.model.optimize() {
            Ok(_) => (),
            Err(error) => panic!("Could not solve the model: {:?}", error)
        };
    }

    fn get_solution(&self, seq_id: usize) -> Array1<usize> {
        let mut sol = Array1::zeros(self.sequences[seq_id].len());
        let vs = &self.vars[seq_id];
        let mut state_from = 0;
        for t in 0..sol.len() {
            for state in 0..self.hmm.nstates() {
                let arc = Arc {state_from, state_to: state, time_to: t};
                match vs.get(&arc) {
                    Some(v) => {
                        let value = self.model.get_values(attr::X, &[v.clone()]).unwrap()[0];
                        if value == 1.0 {
                            sol[t] = state;
                            state_from = state;
                            continue;
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
