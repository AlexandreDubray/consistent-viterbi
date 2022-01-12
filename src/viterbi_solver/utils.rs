use super::super::hmm::hmm::HMM;
use super::constraints::Constraints;
use ndarray::prelude::*;
use rand::prelude::*;

use std::ops::Index;

#[derive(Debug, Copy, Clone)]
pub struct MetaElements<const D: usize> {
    pub seq: usize,
    pub t: usize,
    pub value: [usize; D],
    pub constraint_component: i32,
    active_cstr: bool,
    pub last_of_constraint: bool,
}

impl<const D: usize> MetaElements<D> {

    pub fn new(seq: usize, t: usize, value: [usize; D], constraint_component: i32, active_cstr: bool, last_of_constraint: bool) -> Self {
        Self { seq, t, value, constraint_component, active_cstr, last_of_constraint}
    }

    pub fn arc_p(&self, hmm: &HMM<D>, state_from: usize, state: usize) -> f64 {
        if self.t == 0 {
            hmm.init_prob(state, self.value)
        } else {
            hmm.transition_prob(state_from, state, self.value)
        }
    }

    pub fn transitions(&self, hmm: &HMM<D>, state: usize) -> Array1<f64> {
        if self.t == 0 {
            Array1::from_elem(hmm.nstates(), hmm.pi[state])
        } else {
            Array1::from_shape_fn(hmm.nstates(), |s| hmm.a[[s, state]])
        }
    }

    pub fn can_be_emited(&self, hmm: &HMM<D>, state: usize) -> bool {
        hmm.emit_prob(state, self.value) > f64::NEG_INFINITY
    }

    pub fn is_constrained(&self) -> bool {
        self.active_cstr
    }
}

pub struct SuperSequence<'a, const D: usize> {
    sequences: &'a Vec<Vec<[usize; D]>>,
    hmm: &'a HMM<D>,
    constraints: &'a Constraints,
    elements: Array1<MetaElements<D>>,
    super_seq_start: Vec<usize>,
    orig_seq_sizes: Vec<usize>,
    rng: StdRng,
    nb_active_cstr: usize,
}

impl<'b, const D: usize> SuperSequence<'b, D> {

    pub fn from(sequences: &'b Vec<Vec<[usize; D]>>, constraints: &'b Constraints, hmm: &'b HMM<D>) -> Self {
        let size: usize = (0..sequences.len()).map(|x| sequences[x].len()).sum();
        let mut super_seq_start: Vec<usize> = (0..sequences.len()).collect();
        let mut cur_seq = 0;
        let mut cur_t = 0;
        let mut elements: Array1<MetaElements<D>> = (0..size).map(|t| {
            let v = sequences[cur_seq][cur_t];
            let mut cstr = -1;
            for comp_id in 0..constraints.components.len() {
                if constraints.components[comp_id].contains(&(cur_seq, cur_t)) {
                    cstr = comp_id as i32;
                    break;
                }
            }
            let cstr_active = if cstr == -1 { false } else { true };
            let e = MetaElements::new(cur_seq, cur_t, v, cstr, cstr_active, false);
            cur_t += 1;
            if cur_t == sequences[cur_seq].len() {
                cur_seq += 1;
                if cur_seq != sequences.len() {
                    super_seq_start[cur_seq] = t + 1;
                }
                cur_t = 0;
            }
            e
        }).collect();
        let mut seen = Array1::from_elem(constraints.components.len(), false);
        let mut nb_active_cstr = 0;
        for t in (0..elements.len()).rev() {
            if elements[t].is_constrained() {
                let ucomp = elements[t].constraint_component as usize;
                if !seen[ucomp] {
                    nb_active_cstr += 1;
                    seen[ucomp] = true;
                    elements[t].last_of_constraint = true;
                }
            }
        }
        let orig_seq_sizes = (0..sequences.len()).map(|seq_id| sequences[seq_id].len()).collect();
        let rng = StdRng::seed_from_u64(3019);
        Self { sequences, hmm, constraints, elements , super_seq_start, orig_seq_sizes, rng, nb_active_cstr}
    }

    fn get_sequences_ordering(&self) -> Vec<usize> {
        type OrderingReturn = (usize, f64, usize);
        let mut nb_elem_cstr: Vec<OrderingReturn> = (0..self.super_seq_start.len()).map(|seq_id| {
            let start = self.super_seq_start[seq_id];
            let size = self.orig_seq_sizes[seq_id];
            let mut possible_states = 0.0;
            let mut is_constrained = false;
            for t in start..(start + size) {
                is_constrained = self.elements[t].is_constrained();
                for state in 0..self.hmm.nstates() {
                    if self.hmm.emit_prob(state, self.elements[t].value) > f64::NEG_INFINITY {
                        possible_states += 1.0;
                    }
                }
            }
            let average_state = possible_states / size as f64;
            let cstr_weight = if is_constrained { 1 } else { 0 };
            (cstr_weight, average_state, seq_id)
        }).collect();

        nb_elem_cstr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut first_idx_cstr = -1;
        let mut idx = 0;
        let ret = (0..nb_elem_cstr.len()).map(|x| -> usize {
            if first_idx_cstr == -1 && nb_elem_cstr[x].0 == 1 {
                first_idx_cstr = idx as i32;
            }
            idx += self.sequences[nb_elem_cstr[x].2].len();
            nb_elem_cstr[x].2
        }).collect();
        ret
    }

    fn reorder(&mut self) {
        let ordering = self.get_sequences_ordering();
        let mut new_elements = Array1::from_shape_fn(self.elements.len(), |idx| self.elements[idx]);
        let mut idx = 0;
        for seq_id in ordering {
            let start = self.super_seq_start[seq_id];
            self.super_seq_start[seq_id] = idx;
            let size = self.orig_seq_sizes[seq_id];
            for t in start..(start+size) {
                new_elements[idx] = self.elements[t];
                idx += 1;
            }
        }
        let mut seen = Array1::from_elem(self.constraints.components.len(), false);
        self.nb_active_cstr = 0;
        for t in (0..new_elements.len()).rev() {
            if new_elements[t].is_constrained() {
                let ucomp = new_elements[t].constraint_component as usize;
                if !seen[ucomp] {
                    self.nb_active_cstr += 1;
                    seen[ucomp] = true;
                    new_elements[t].last_of_constraint = true;
                }
            }
        }
        self.elements = new_elements;
    }


    pub fn recompute_constraints(&mut self, proportion: f64) {
        for t in 0..self.len() {
            if self.elements[t].constraint_component != -1 && self.rng.gen::<f64>() <= proportion {
                self.elements[t].active_cstr = true;
            } else {
                self.elements[t].active_cstr = false;
            }
        }
        self.reorder();
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn parse_solution(&self, solution: &Array1<usize>) -> Array1<Array1<usize>> {
        let mut sol = Array1::from_iter((0..self.orig_seq_sizes.len()).map(|i| Array1::from_elem(self.orig_seq_sizes[i], 0)));
        for i in 0..self.elements.len() {
            let el = &self.elements[i];
            sol[el.seq][el.t] = solution[i];
        }
        sol
    }

    pub fn is_constrained(&self, idx: usize) -> bool {
        self.elements[idx].constraint_component != -1
    }

    pub fn constraint_size(&self, comp_id: usize) -> usize {
        self.constraints.components[comp_id].len()
    }
    
    pub fn number_constraints(&self) -> usize {
        self.nb_active_cstr
    }
}

impl<const D: usize> Index<usize> for SuperSequence<'_, D> {
    type Output = MetaElements<D>;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.elements[idx]
    }
}
