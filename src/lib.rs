use std::ops::Range;

use layer::Layer;
use nll::Nll;
use relu::Relu;

mod layer;
pub mod mnist;
mod nll;
mod relu;

#[cfg(test)]
mod tests;

pub trait Train {
    const INPUT_SIZE: usize;
    const OUTPUT_SIZE: usize;
    const BATCH_SIZE: usize;
    const EPOCHS: usize;

    fn train_data(&self, r: Range<usize>) -> &[f64];
    fn train_labels(&self, r: Range<usize>) -> &[u8];
    fn test_data(&self) -> &[f64];
    fn test_labels(&self) -> &[u8];
}

pub fn train<T>(set: T) -> Vec<f64>
where
    T: Train,
{
    let mut results = Vec::with_capacity(T::EPOCHS);

    let mut layer1 = Layer::new(T::INPUT_SIZE, 100);
    let mut relu1 = Relu::new();
    let mut layer2 = Layer::new(100, T::OUTPUT_SIZE);

    for e in 0..T::EPOCHS {
        // training
        for i in 0..60_000 / T::BATCH_SIZE {
            let inputs = set.train_data(
                i * T::INPUT_SIZE * T::BATCH_SIZE
                    ..(i + 1) * T::INPUT_SIZE * T::BATCH_SIZE,
            );
            let targets =
                &set.train_labels(i * T::BATCH_SIZE..(i + 1) * T::BATCH_SIZE);

            // Go forward and get loss
            let outputs1 = layer1.forward(inputs.to_vec());
            let outputs2 = relu1.forward(outputs1);
            let outputs3 = layer2.forward(outputs2);
            let loss = Nll::new(T::OUTPUT_SIZE, outputs3, targets);

            // Update network
            let grads1 = layer2.backward(loss.input_grads);
            let grads2 = relu1.backward(grads1.input_grads);
            let grads3 = layer1.backward(grads2);
            layer1.apply_gradients(grads3.weight_grads);
            layer2.apply_gradients(grads1.weight_grads);
        }

        // validation
        let mut correct = 0;
        let outputs1 = layer1.forward(set.test_data().to_vec());
        let outputs2 = relu1.forward(outputs1);
        let outputs3 = layer2.forward(outputs2);

        for b in 0..10_000 {
            let mut max_guess = outputs3[b * T::OUTPUT_SIZE];
            let mut guess_index = 0;
            for (i, o) in outputs3[b * T::OUTPUT_SIZE..(b + 1) * T::OUTPUT_SIZE]
                .iter()
                .enumerate()
            {
                if *o > max_guess {
                    max_guess = *o;
                    guess_index = i;
                }
            }
            if guess_index as u8 == set.test_labels()[b] {
                correct += 1;
            }
        }

        let res = correct as f64 / 100.0;
        println!("{e:5} average accuracy {:.2}", res);

        results.push(res);
    }

    results
}
