use dnnosaur::{qff::Qff, Train};

fn main() {
    // mnist::Data::read_mnist().train(25);
    Qff::default().load("qff_data").unwrap().train(200);
}
