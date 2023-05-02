#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use layer::Layer;
use nll::Nll;
use relu::Relu;

mod layer;
mod mnist;
mod nll;
mod relu;

const INPUT_SIZE: usize = 784;
const OUTPUT_SIZE: usize = 10;
const BATCH_SIZE: usize = 32;
const EPOCHS: usize = 25;

fn main() {
    let mnist_data = mnist::Data::read_mnist();

    // let loss_function: nll::Nll<OUTPUT_SIZE> = nll::Nll::new();

    let mut layer1: Layer<INPUT_SIZE, 100> = Layer::new();
    let mut relu1 = Relu::new();
    let mut layer2: Layer<100, OUTPUT_SIZE> = Layer::new();

    for e in 0..EPOCHS {
        // training
        for i in 0..60_000 / BATCH_SIZE {
            let inputs = &mnist_data.train_images[i * INPUT_SIZE * BATCH_SIZE
                ..(i + 1) * INPUT_SIZE * BATCH_SIZE];
            let targets =
                &mnist_data.train_labels[i * BATCH_SIZE..(i + 1) * BATCH_SIZE];

            // Go forward and get loss
            let outputs1 = layer1.forward(inputs.to_vec());
            let outputs2 = relu1.forward(outputs1);
            let outputs3 = layer2.forward(outputs2);
            let loss = Nll::<OUTPUT_SIZE>::nll(outputs3, targets.to_vec());

            // Update network
            let grads1 = layer2.backward(loss.input_grads);
            let grads2 = relu1.backward(grads1.input_grads);
            let grads3 = layer1.backward(grads2);
            layer1.apply_gradients(grads3.weight_grads);
            layer2.apply_gradients(grads1.weight_grads);
        }

        // validation
        let mut correct = 0;
        let outputs1 = layer1.forward(mnist_data.test_images.to_vec());
        let outputs2 = relu1.forward(outputs1);
        let outputs3 = layer2.forward(outputs2);

        for b in 0..10_000 {
            let mut max_guess = outputs3[b * OUTPUT_SIZE];
            let mut guess_index = 0;
            for (i, o) in outputs3[b * OUTPUT_SIZE..(b + 1) * OUTPUT_SIZE]
                .iter()
                .enumerate()
            {
                if *o > max_guess {
                    max_guess = *o;
                    guess_index = i;
                }
            }
            if guess_index as u8 == mnist_data.test_labels[b] {
                correct += 1;
            }
        }

        println!("{e:5} average accuracy {:.2}", correct as f64 / 100.0);
    }
}
