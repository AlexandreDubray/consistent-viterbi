use ndarray::Array1;

pub mod constraints;
pub mod opti;
pub mod dp;
pub mod utils;
pub mod viterbi;
pub mod cp;
pub mod cfn;

pub trait Solver {
    fn solve(&mut self);
    fn get_solution(&self) -> &Array1<usize>;
    fn get_objective(&self) -> f64;
    fn get_name(&self) -> String;
}
