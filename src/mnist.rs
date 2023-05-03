use std::io::{Read, Seek};

use crate::{nll::NllOutput, Train};

#[derive(Debug)]
pub struct Data {
    pub training_data: Vec<f64>,
    pub train_labels: Vec<u8>,
    pub test_images: Vec<f64>,
    pub test_labels: Vec<u8>,
}

impl Data {
    pub fn read_mnist() -> Self {
        let bytes = Self::read_idx_file("data/train-images-idx3-ubyte", 16);
        assert_eq!(Self::INPUT_SIZE * 60_000, bytes.len());
        let train_images = bytes.iter().map(|&b| b as f64 / 255.0).collect();

        let train_labels =
            Self::read_idx_file("data/train-labels-idx1-ubyte", 8);

        let bytes = Self::read_idx_file("data/t10k-images-idx3-ubyte", 16);
        assert_eq!(Self::INPUT_SIZE * 10_000, bytes.len());
        let test_images = bytes.iter().map(|&b| b as f64 / 255.0).collect();

        let test_labels = Self::read_idx_file("data/t10k-labels-idx1-ubyte", 8);

        Self {
            training_data: train_images,
            train_labels,
            test_images,
            test_labels,
        }
    }

    fn read_idx_file(path: &str, skip: u64) -> Vec<u8> {
        let mut reader = std::fs::File::open(path).unwrap();
        reader.seek(std::io::SeekFrom::Start(skip)).unwrap();
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).unwrap();
        buf
    }
}

impl Train<u8> for Data {
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

    fn check_output(&self, outputs3: Vec<f64>) -> i32 {
        let mut correct = 0;
        for b in 0..10_000 {
            let guess_index = outputs3
                [b * Self::OUTPUT_SIZE..(b + 1) * Self::OUTPUT_SIZE]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .expect("failed to find a max somehow")
                .0;
            if guess_index as u8 == self.test_labels()[b] {
                correct += 1;
            }
        }
        correct
    }

    fn nll(n: usize, inputs: Vec<f64>, targets: &[u8]) -> NllOutput {
        let batch_size = targets.len();
        let mut sum_e = vec![0.0; batch_size];
        for b in 0..batch_size {
            let mut sum = 0.0;
            for i in 0..n {
                sum += inputs[b * n + i].exp();
            }
            sum_e[b] = sum;
        }

        let mut loss = vec![0.0; batch_size];
        for b in 0..batch_size {
            loss[b] = -1.0
                * (inputs[b * n + targets[b] as usize].exp() / sum_e[b]).ln();
        }

        let mut input_grads = vec![0.0; batch_size * n];
        for b in 0..batch_size {
            for i in 0..n {
                input_grads[b * n + i] = inputs[b * n + i].exp() / sum_e[b];
                if i == targets[b] as usize {
                    input_grads[b * n + i] -= 1.0;
                }
            }
        }

        NllOutput {
            n,
            loss,
            input_grads,
        }
    }

    const INPUT_SIZE: usize = 784;
    const OUTPUT_SIZE: usize = 10;
    const BATCH_SIZE: usize = 32;
    const LABEL_SIZE: usize = 1;
}
