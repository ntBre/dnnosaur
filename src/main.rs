use dnnosaur::{qff::Qff, LossFn, Train};

fn main() {
    // mnist::Data::read_mnist().train(25);
    Qff::load("qff_data").unwrap().train(200, LossFn::Sigmoid);
}
