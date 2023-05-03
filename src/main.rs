use dnnosaur::{mnist, Train};

fn main() {
    mnist::Data::read_mnist().train(25);
}
