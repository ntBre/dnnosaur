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

    fn train_data(&self, r: Range<usize>) -> &[f64];
    fn train_labels(&self, r: Range<usize>) -> &[u8];
    fn test_data(&self) -> &[f64];
    fn test_labels(&self) -> &[u8];

    fn train(&self, epochs: usize) -> Vec<f64> {
        let mut results = Vec::with_capacity(epochs);

        let mut layer1 = Layer::new(Self::INPUT_SIZE, 100);
        let mut relu1 = Relu::new();
        let mut layer2 = Layer::new(100, Self::OUTPUT_SIZE);

        for e in 0..epochs {
            // training
            for i in 0..60_000 / Self::BATCH_SIZE {
                let inputs = self.train_data(
                    i * Self::INPUT_SIZE * Self::BATCH_SIZE
                        ..(i + 1) * Self::INPUT_SIZE * Self::BATCH_SIZE,
                );
                let targets = &self.train_labels(
                    i * Self::BATCH_SIZE..(i + 1) * Self::BATCH_SIZE,
                );

                // Go forward and get loss
                let outputs1 = layer1.forward(inputs.to_vec());
                let outputs2 = relu1.forward(outputs1);
                let outputs3 = layer2.forward(outputs2);
                let loss = Nll::new(Self::OUTPUT_SIZE, outputs3, targets);

                // Update network
                let grads1 = layer2.backward(loss.input_grads);
                let grads2 = relu1.backward(grads1.input_grads);
                let grads3 = layer1.backward(grads2);
                layer1.apply_gradients(grads3.weight_grads);
                layer2.apply_gradients(grads1.weight_grads);
            }

            // validation
            let mut correct = 0;
            let outputs1 = layer1.forward(self.test_data().to_vec());
            let outputs2 = relu1.forward(outputs1);
            let outputs3 = layer2.forward(outputs2);

            for b in 0..10_000 {
                let mut max_guess = outputs3[b * Self::OUTPUT_SIZE];
                let mut guess_index = 0;
                for (i, o) in outputs3
                    [b * Self::OUTPUT_SIZE..(b + 1) * Self::OUTPUT_SIZE]
                    .iter()
                    .enumerate()
                {
                    if *o > max_guess {
                        max_guess = *o;
                        guess_index = i;
                    }
                }
                if guess_index as u8 == self.test_labels()[b] {
                    correct += 1;
                }
            }

            let res = correct as f64 / 100.0;
            println!("{e:5} average accuracy {:.2}", res);

            results.push(res);
        }

        results
    }
}
