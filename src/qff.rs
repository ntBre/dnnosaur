use crate::NllOutput;
use crate::Train;

struct Qff {
    pub training_data: Vec<f64>,
    pub train_labels: Vec<u8>,
    pub test_images: Vec<f64>,
    pub test_labels: Vec<u8>,
}

#[allow(unused)]
impl Train<u8> for Qff {
    // not sure what these are yet. I think it will depend on the maximum length
    // of my training data
    const INPUT_SIZE: usize = 0;
    const LABEL_SIZE: usize = 0;
    const OUTPUT_SIZE: usize = 0;

    // leaving this the same as before, not sure what to do here really
    const BATCH_SIZE: usize = 32;

    fn train_data(&self, r: std::ops::Range<usize>) -> &[f64] {
        &self.training_data[r]
    }

    fn train_labels(&self, r: std::ops::Range<usize>) -> &[u8] {
        &self.train_labels[r]
    }

    fn test_data(&self) -> &[f64] {
        &self.test_images
    }

    fn test_labels(&self) -> &[u8] {
        &self.test_labels
    }

    fn nll(n: usize, inputs: Vec<f64>, targets: &[u8]) -> NllOutput {
        todo!()
    }

    fn check_output(&self, outputs3: Vec<f64>) -> i32 {
        todo!()
    }
}
