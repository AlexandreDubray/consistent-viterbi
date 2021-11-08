use super::hmm::HMM;
use super::constraints::Constraints;
use ndarray::Array1;

use std::collections::HashSet;

use std::ops::Index;

#[derive(Debug)]
pub struct MetaElements {
    pub seq: usize,
    pub t: usize,
    pub value: usize,
    pub constraint_component: i32,
    pub last_of_constraint: bool,
    pub previous_idx_cstr: Option<usize>
}

impl MetaElements {

    pub fn new(seq: usize, t: usize, value: usize, constraint_component: i32, last_of_constraint: bool, previous_idx_cstr: Option<usize>) -> Self {
        Self { seq, t, value, constraint_component, last_of_constraint, previous_idx_cstr}
    }

    pub fn arc_p(&self, hmm: &HMM, state_from: usize, state: usize) -> f64 {
        if self.t == 0 {
            hmm.init_prob(state, self.value)
        } else {
            hmm.transition_prob(state_from, state, self.value)
        }
    }

    pub fn can_be_emited(&self, hmm: &HMM, state: usize) -> bool {
        hmm.emit_prob(state, self.value) > f64::NEG_INFINITY
    }
}

pub struct SuperSequence {
    elements: Vec<MetaElements>,
    pub nb_cstr: usize,
    pub nb_seqs: usize,
    pub first_pos_cstr: Array1<usize>,
    pub seq_sizes: Array1<usize>,
    pub seq_starts: Array1<usize>,
    pub seq_per_cstr: Array1<HashSet<usize>>,
    orig_seq_sizes: Array1<usize>
}

impl SuperSequence {

    fn get_sequences_ordering(hmm: &HMM, sequences: &Array1<Array1<usize>>) -> Vec<usize> {
        type OrderingReturn = (f64, usize);
        let mut nb_elem_cstr: Vec<OrderingReturn> = (0..sequences.len()).map(|seq_id| -> OrderingReturn {
            let mut possible_states = 0.0;
            for t in 0..sequences[seq_id].len() {
                for state in 0..hmm.nstates() {
                    if hmm.emit_prob(state, sequences[seq_id][t]) > f64::NEG_INFINITY {
                        possible_states += 1.0;
                    }
                }
            }
            let average_state = possible_states / sequences[seq_id].len() as f64;
            (average_state, seq_id)
        }).collect();
        nb_elem_cstr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (0..nb_elem_cstr.len()).map(|x| -> usize { nb_elem_cstr[x].1 }).collect()
    }

    pub fn from(sequences: &Array1<Array1<usize>>, constraints: &Constraints, hmm: &HMM) -> Self {
        let size: usize = (0..sequences.len()).map(|x| sequences[x].len()).sum();
        let nb_cstr = constraints.components.len();
        let nb_seqs = sequences.len();

        let mut first_pos_cstr = Array1::from_elem(nb_cstr, 0);
        let mut seen = Array1::from_elem(nb_cstr, false);

        let mut elements: Vec<MetaElements> = Vec::with_capacity(size);

        let ordering = SuperSequence::get_sequences_ordering(hmm, sequences);
        let mut seq_starts = Array1::from_elem(sequences.len(), 0);
        let mut seq_sizes = Array1::from_elem(sequences.len(), 0);

        let mut seq_per_cstr: Array1<HashSet<usize>> = Array1::from_elem(nb_cstr, HashSet::new());

        let mut last_seen_cstr: Array1<Option<usize>> = Array1::from_elem(nb_cstr, None);
        let mut idx = 0;
        for seq_id in ordering {
            let sequence = &sequences[seq_id];
            seq_starts[idx] = elements.len();
            seq_sizes[idx] = sequence.len();
            for t in 0..sequence.len() {
                let comp_id = constraints.get_comp_id(seq_id, t);
                if comp_id != -1 && !seen[comp_id as usize] {
                    seen[comp_id as usize] = true;
                    first_pos_cstr[comp_id as usize] = elements.len();
                }
                if comp_id != -1 {
                    seq_per_cstr[comp_id as usize].insert(idx);
                }
                let previous_idx_cstr = if comp_id == -1 {
                    None
                } else {
                    let ret = last_seen_cstr[comp_id as usize];
                    last_seen_cstr[comp_id as usize] = Some(idx);
                    ret
                };
                let element = MetaElements::new(seq_id, t, sequence[t], comp_id, false, previous_idx_cstr);
                elements.push(element);
            }
            idx += 1;
        }

        seen.fill(false);
        let count = 0;
        for i in (0..size).rev() {
            if count == nb_cstr {
                break;
            }
            let elem = &mut elements[i];
            if elem.constraint_component != -1 {
                let ucomp = elem.constraint_component as usize;
                if !seen[ucomp] {
                    seen[ucomp] = true;
                    elem.last_of_constraint = true;
                }
            }
        }

        let orig_seq_sizes = (0..sequences.len()).map(|seq_id| sequences[seq_id].len()).collect();
        Self { elements , nb_cstr: constraints.components.len(), nb_seqs, first_pos_cstr, seq_sizes, seq_starts, seq_per_cstr, orig_seq_sizes}
    }

    pub fn recompute_constraints(&mut self, constraints: &Constraints) {
        self.nb_cstr = constraints.components.len();
        let mut first_pos_cstr = Array1::from_elem(self.nb_cstr, 0);
        let mut seen = Array1::from_elem(self.nb_cstr, false);
        let mut seq_per_cstr: Array1<HashSet<usize>> = Array1::from_elem(self.nb_cstr, HashSet::new());
        let mut idx = 0;
        let T = self.elements.len();
        for t in 0..T {
            let element = &mut self.elements[t];
            element.constraint_component = constraints.get_comp_id(element.seq, element.t);
            if element.constraint_component != -1 {
                let ucomp = element.constraint_component as usize;
                if !seen[ucomp] {
                    seen[ucomp] = true;
                    first_pos_cstr[ucomp] = t;
                }
                seq_per_cstr[ucomp].insert(idx);
            }
            if t < T-1 && element.seq != self.elements[t+1].seq {
                idx += 1;
            }
        }
        self.first_pos_cstr = first_pos_cstr;
        self.seq_per_cstr = seq_per_cstr;
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn parse_solution(&self, solution: &Array1<usize>) -> Array1<Array1<usize>> {
        let mut sol = Array1::from_iter((0..self.seq_sizes.len()).map(|i| Array1::from_elem(self.orig_seq_sizes[i], 0)));
        for i in 0..self.elements.len() {
            let el = &self.elements[i];
            sol[el.seq][el.t] = solution[i];
        }
        sol
    }
}

impl Index<usize> for SuperSequence {
    type Output = MetaElements;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.elements[idx]
    }
}
