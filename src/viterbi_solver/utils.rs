use super::super::hmm::hmm::HMM;
use super::constraints::Constraints;
use ndarray::prelude::*;

use std::ops::Index;

#[derive(Debug)]
pub struct MetaElements<const D: usize> {
    pub seq: usize,
    pub t: usize,
    pub value: [usize; D],
    pub constraint_component: i32,
    pub last_of_constraint: bool,
    pub previous_idx_cstr: Option<usize>
}

impl<const D: usize> MetaElements<D> {

    pub fn new(seq: usize, t: usize, value: [usize; D], constraint_component: i32, last_of_constraint: bool, previous_idx_cstr: Option<usize>) -> Self {
        Self { seq, t, value, constraint_component, last_of_constraint, previous_idx_cstr}
    }

    pub fn empty() -> Self {
        Self { seq: 0, t: 0, value: [0; D], constraint_component: 0, last_of_constraint: false, previous_idx_cstr: None}
    }

    pub fn arc_p(&self, hmm: &HMM<D>, state_from: usize, state: usize) -> f64 {
        if self.t == 0 {
            hmm.init_prob(state, self.value)
        } else {
            hmm.transition_prob(state_from, state, self.value)
        }
    }

    pub fn can_be_emited(&self, hmm: &HMM<D>, state: usize) -> bool {
        hmm.emit_prob(state, self.value) > f64::NEG_INFINITY
    }
}

pub struct SuperSequence<'a, const D: usize> {
    sequences: &'a Vec<Vec<[usize; D]>>,
    hmm: &'a HMM<D>,
    constraints: &'a mut Constraints,
    elements: Array1<MetaElements<D>>,
    pub nb_cstr: usize,
    pub nb_seqs: usize,
    pub first_pos_cstr: Array1<usize>,
    orig_seq_sizes: Array1<usize>,
    pub active_constraints: Array1<bool>
}

impl<'b, const D: usize> SuperSequence<'b, D> {

    pub fn from(sequences: &'b Vec<Vec<[usize; D]>>, constraints: &'b mut Constraints, hmm: &'b HMM<D>) -> Self {
        let size: usize = (0..sequences.len()).map(|x| sequences[x].len()).sum();
        let nb_cstr = constraints.components.len();
        let nb_seqs = sequences.len();

        let elements: Array1<MetaElements<D>> = (0..size).map(|_| MetaElements::empty()).collect();
        let first_pos_cstr = Array1::from_elem(1, 0);

        let orig_seq_sizes = (0..sequences.len()).map(|seq_id| sequences[seq_id].len()).collect();
        let active_constraints = Array1::from_elem(nb_cstr, true);
        Self { sequences, hmm, constraints, elements , nb_cstr, nb_seqs, first_pos_cstr, orig_seq_sizes, active_constraints }
    }

    fn get_sequences_ordering(&self) -> Vec<usize> {
        type OrderingReturn = (usize, f64, usize);
        let mut nb_unconstrained = 0;
        let mut nb_elem_cstr: Vec<OrderingReturn> = (0..self.sequences.len()).map(|seq_id| -> OrderingReturn {
            let mut possible_states = 0.0;
            let mut is_constrained = false;
            for t in 0..self.sequences[seq_id].len() {
                let comp_id = self.constraints.get_comp_id(seq_id, t);
                is_constrained |= comp_id != -1 && self.active_constraints[comp_id as usize];
                for state in 0..self.hmm.nstates() {
                    if self.hmm.emit_prob(state, self.sequences[seq_id][t]) > f64::NEG_INFINITY {
                        possible_states += 1.0;
                    }
                }
            }
            let average_state = possible_states / self.sequences[seq_id].len() as f64;
            let cstr_weight = if is_constrained { 1 } else { nb_unconstrained += 1; 0 };
            (cstr_weight, average_state, seq_id)
        }).collect();
        println!("Number of unconstrained sequences {} (out of {})", nb_unconstrained, self.sequences.len());
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

    pub fn reorder(&mut self) {
        let ordering = self.get_sequences_ordering();
        let mut last_seen_cstr: Array1<Option<usize>> = Array1::from_elem(self.nb_cstr, None);
        self.first_pos_cstr = Array1::from_elem(self.nb_cstr, 0);
        let mut idx = 0;
        for seq_id in ordering {
            let sequence = &self.sequences[seq_id];
            for t in 0..sequence.len() {
                let comp_id = self.constraints.get_comp_id(seq_id, t);
                let elem = &mut self.elements[idx];
                if comp_id != -1 && self.active_constraints[comp_id as usize] {
                    let ucomp = comp_id as usize;
                    match last_seen_cstr[ucomp] {
                        None => {
                            self.first_pos_cstr[ucomp] = idx;
                            elem.previous_idx_cstr = None;
                        },
                        Some(prev_idx) => {
                            elem.previous_idx_cstr = Some(prev_idx);
                        }
                    };
                    last_seen_cstr[ucomp] = Some(idx);
                }
                elem.seq = seq_id;
                elem.t = t;
                elem.value = sequence[t];
                elem.constraint_component = comp_id;
                idx += 1;
            }
        }
    }


    pub fn recompute_constraints(&mut self, proportion: f64) {
        self.constraints.keep_prop(proportion);
        self.nb_cstr = self.constraints.components.len();
        self.active_constraints = Array1::from_elem(self.nb_cstr, true);
        let mut first_pos_cstr = Array1::from_elem(self.nb_cstr, 0);
        let mut seen = Array1::from_elem(self.nb_cstr, false);
        let length = self.elements.len();
        for t in 0..length {
            let element = &mut self.elements[t];
            element.constraint_component = self.constraints.get_comp_id(element.seq, element.t);
            if element.constraint_component != -1 {
                let ucomp = element.constraint_component as usize;
                if !seen[ucomp] {
                    seen[ucomp] = true;
                    first_pos_cstr[ucomp] = t;
                }
            }
        }
        self.first_pos_cstr = first_pos_cstr;
        self.active_constraints = Array1::from_elem(self.nb_cstr, true);
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

    pub fn deactive_constraints(&mut self) {
        self.active_constraints.fill(false);
    }

    pub fn activate_constraints(&mut self) {
        self.active_constraints.fill(true);
    }

    pub fn activate_constraint(&mut self, comp_id: usize) {
        self.active_constraints[comp_id] = true;
    }

    pub fn is_constrained(&self, idx: usize) -> bool {
        let element = &self.elements[idx];
        element.constraint_component != -1 && self.active_constraints[element.constraint_component as usize]
    }

    pub fn constraint_size(&self, comp_id: usize) -> usize {
        self.constraints.components[comp_id].len()
    }
}

impl<const D: usize> Index<usize> for SuperSequence<'_, D> {
    type Output = MetaElements<D>;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.elements[idx]
    }
}
