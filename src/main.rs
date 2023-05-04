use dnnosaur::{qff::Qff, Train};

fn main() {
    // mnist::Data::read_mnist().train(25);
    Qff::load("lxm").unwrap().train(15);
}
