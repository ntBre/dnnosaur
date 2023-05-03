use dnnosaur::{mnist, train};

fn main() {
    train(mnist::Data::read_mnist(), 25);
}
