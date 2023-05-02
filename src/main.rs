#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

mod layer;
mod mnist;
mod nll;
mod relu;

const INPUT_SIZE: usize = 784;
const OUTPUT_SIZE: usize = 10;
const BATCH_SIZE: usize = 32;
const EPOCHS: usize = 25;

fn main() {
    let data = mnist::Data::read_mnist();
}
