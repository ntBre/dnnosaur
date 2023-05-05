use std::io::Write;
use std::{fs::File, ops::Range};

use layer::Layer;
use nll::NllOutput;
use relu::Relu;

mod layer;
pub mod mnist;
pub mod qff;

mod nll;
mod relu;

#[cfg(test)]
mod tests;

pub trait Train<Label> {
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn batch_size(&self) -> usize;

    /// the total size of the training set
    fn data_size(&self) -> usize;

    /// the length of each label value
    fn label_size(&self) -> usize;

    /// returns a slice of the training data
    fn train_data(&self, r: Range<usize>) -> &[f64];

    /// returns a slice of the training labels
    fn train_labels(&self, r: Range<usize>) -> &[Label];

    /// returns a reference to the validation data
    fn test_data(&self) -> &[f64];

    /// returns a reference to the validation labels
    fn test_labels(&self) -> &[Label];

    /// assess the performance of the current output of the model
    fn check_output(&self, got: &[f64], want: &[Label]) -> f64;

    /// the loss function for the model
    fn nll(&self, inputs: Vec<f64>, targets: &[Label]) -> NllOutput;

    /// perform the actual training
    fn train(&self, epochs: usize) -> Vec<f64> {
        let mut results = Vec::with_capacity(epochs);

        const EDGES: usize = 100;

        let mut layer1 = Layer::new(self.input_size(), EDGES);
        let mut relu1 = Relu::new();
        let mut layer2 = Layer::new(EDGES, self.output_size());

        let mut output_log = File::create("train.log").unwrap();
        let mut accuracy_log = File::create("accuracy.log").unwrap();

        for e in 0..epochs {
            // training
            let mut pred_error = 0.0;
            for i in 0..self.data_size() / self.batch_size() {
                let inputs = self.train_data(
                    i * self.input_size() * self.batch_size()
                        ..(i + 1) * self.input_size() * self.batch_size(),
                );
                let targets = &self.train_labels(
                    i * self.label_size() * self.batch_size()
                        ..(i + 1) * self.label_size() * self.batch_size(),
                );

                // Go forward and get loss
                let outputs1 = layer1.forward(inputs.to_vec());
                let outputs2 = relu1.forward(outputs1);
                let outputs3 = layer2.forward(outputs2);
                pred_error += self.check_output(&outputs3, targets);
                let loss = self.nll(outputs3, targets);

                // Update network
                let grads1 = layer2.backward(loss.input_grads);
                let grads2 = relu1.backward(grads1.input_grads);
                let grads3 = layer1.backward(grads2);
                layer1.apply_gradients(grads3.weight_grads);
                layer2.apply_gradients(grads1.weight_grads);
            }

            // validation
            let outputs1 = layer1.forward(self.test_data().to_vec());
            let outputs2 = relu1.forward(outputs1);
            let outputs3 = layer2.forward(outputs2);

            writeln!(output_log, "{outputs3:#?}").unwrap();

            let res = self.check_output(&outputs3, self.test_labels());

            println!("{e:5} average accuracy {:.2}", res);
            writeln!(
                accuracy_log,
                "{e:5} {:8.2} {:8.2}",
                res,
                pred_error / (self.data_size() / self.batch_size()) as f64
            )
            .unwrap();

            results.push(res);
        }

        results
    }
}
