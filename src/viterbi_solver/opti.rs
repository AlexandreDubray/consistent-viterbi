use gurobi::*;
use ndarray::Array1;
use ndarray_stats::QuantileExt;
use std::collections::HashMap;

use super::hmm::HMM;
use super::constraints::Constraints;

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Arc {
    state_from: usize,
    state_to: usize,
    time_to: usize
}

/*
struct GlobalOpti {
    hmm: HMM,
    sequences: Array1<Array1<usize>>,
    constraints: Constraints
}

impl GlobalOpti {

    pub fn new(hmm: &HMM, sequences: &Array1<Array1<usize>>, constraints: &Constraints) -> Self {
        Self {hmm, sequences, constraints }
    }
}
*/

fn get_outflow(seq_len: usize, t: usize, state: usize, var_map: &HashMap<Arc, Var>) -> LinExpr {
    let mut outflow = LinExpr::new();
    if t == seq_len - 1 {
        let arc = Arc { state_from: state, state_to: 0, time_to: t+1 };
        match var_map.get(&arc) {
            Some(v) => outflow = outflow.add_term(1.0, v.clone()),
            None => ()
        };
    } else {

    }

    outflow
}

pub fn create_model(hmm: HMM, sequences: Array1<Array1<usize>>, constraints: Constraints) -> Model {

    println!("Creating the global optimisation problem");
    let env = Env::new("logfile.log").unwrap();
    let mut model = Model::new("model", &env).unwrap();

    let mut vars = sequences.map(|_| -> HashMap<Arc, Var> { HashMap::new() });

    println!("Adding the variables in the model");
    for i in 0..sequences.len() {
        if i % 10 == 0 {
            println!("{}/{}", i, sequences.len());
        }
        let sequence = &sequences[i];
        let mut source_flow = LinExpr::new();
        let mut target_flow = LinExpr::new();
        for t in 0..sequence.len() + 1 {

            if t == 0 {
                let p = hmm.init_prob(sequence[t]);
                for state in 0..hmm.nstates() {
                    let proba = *p.get(state).unwrap();
                    if p[i] > f64::NEG_INFINITY {
                        let arc = Arc { state_from: 0, state_to: state, time_to: t };
                        let x = model.add_var("", Binary, proba, 0.0, 1.0, &[], &[]).unwrap();
                        source_flow = source_flow.add_term(1.0, x.clone());
                        vars[i].insert(arc, x);
                    }
                }
            } else if t == sequence.len() {
                for state in 0..hmm.nstates() {
                    let arc = Arc {state_from: state, state_to: 0, time_to: t};
                    let p = 0.0;
                    let x = model.add_var("", Binary, p, 0.0, 1.0, &[], &[]).unwrap();
                    target_flow = target_flow.add_term(1.0, x.clone());
                    vars[i].insert(arc, x);
                }
            } else {
                for state in 0..hmm.nstates() {
                    let p = hmm.transition(state);
                    let state_from = p.argmax().unwrap();
                    let proba = p[state_from] + hmm.emit_prob(state, sequence[t]);
                    if proba > f64::NEG_INFINITY {
                        let arc = Arc {state_from, state_to: state, time_to:t};
                        let x = model.add_var("", Binary, proba, 0.0, 1.0, &[], &[]).unwrap();
                        vars[i].insert(arc, x);
                    }
                    
                }
            }
        }
        match model.add_constr("", source_flow, Equal, 1.0) {
            Ok(_) => (),
            Err(error) => panic!("Can not add constraint to the model: {:?}", error)
        };
        match model.add_constr("", target_flow, Equal, 1.0) {
            Ok(_) => (),
            Err(error) => panic!("Can not add constraint to the model: {:?}", error)
        };
    }

    println!("Adding the shortest path constraints");

    for i in 0..sequences.len() {
        let sequence = &sequences[i];
        for t in 0..sequence.len() {
            for state in 0..hmm.nstates() {
                let mut flow = LinExpr::new();

                for other_state in 0..hmm.nstates() {
                    let arc_from = Arc{state_from: other_state, state_to: state, time_to: t};
                    let arc_to = Arc{state_from: other_state, state_to: state, time_to: t+1};
                    match vars[i].get(&arc_from) {
                        Some(v) => flow = flow.add_term(1.0, v.clone()),
                        None => ()
                    };
                    match vars[i].get(&arc_to) {
                        Some(v) => flow = flow.add_term(-1.0, v.clone()),
                        None => ()
                    };
                }

                match model.add_constr("", flow, Equal, 0.0) {
                    Ok(_) => (),
                    Err(error) => panic!("Cannot add constraint to the model: {:?}", error)
                };
            }
        }
        if i % 10 == 0 {
            println!("{}/{}", i, sequences.len());
        }
    }

    println!("Adding the consistency constraints");
    for component in constraints.components {
        for i in 0..component.len() {
            for j in i+1..component.len() {
            }
        }
    }

    println!("updating model");
    match model.update() {
        Ok(_) => (),
        Err(error) => panic!("Can not update the model: {:?}", error)
    };
    println!("Writing model");
    model.write("global_opti.lp").unwrap();

    model
}

