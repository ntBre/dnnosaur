use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

use crate::NllOutput;
use crate::Train;

pub struct Qff {
    pub train_data: Vec<f64>,
    pub train_labels: Vec<f64>,
    pub test_images: Vec<f64>,
    pub test_labels: Vec<f64>,
}

impl Qff {
    /// load a [Qff] from the file specified by `p`. This file should contain
    /// the frequencies on the first line, followed by the lxm matrix
    pub fn load(p: impl AsRef<Path>) -> std::io::Result<Self> {
        let f = File::open(p)?;
        let mut lines = BufReader::new(f).lines().flatten();
        let freqs: Vec<f64> = lines
            .next()
            .expect("no lines found")
            .split_ascii_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let mut lxm: Vec<f64> = Vec::new();
        for line in lines {
            lxm.extend(
                line.split_ascii_whitespace()
                    .map(|s| s.parse::<f64>().unwrap()),
            );
        }
        Ok(Self {
            train_data: lxm.clone(),
            train_labels: freqs.clone(),
            test_images: lxm,
            test_labels: freqs,
        })
    }
}

#[allow(unused)]
impl Train<f64> for Qff {
    /// the maximum size of the input lxm matrices
    const INPUT_SIZE: usize = 1296;

    /// the maximum size of the output frequencies - 30 again for benzene
    const LABEL_SIZE: usize = 30;

    /// also the maximum size of the output frequencies - 30 for benzene alone
    const OUTPUT_SIZE: usize = 30;

    const BATCH_SIZE: usize = 1;

    const DATA_SIZE: usize = 1;

    fn train_data(&self, r: std::ops::Range<usize>) -> &[f64] {
        &self.train_data[r]
    }

    fn train_labels(&self, r: std::ops::Range<usize>) -> &[f64] {
        &self.train_labels[r]
    }

    fn test_data(&self) -> &[f64] {
        &self.test_images
    }

    fn test_labels(&self) -> &[f64] {
        &self.test_labels
    }

    fn nll(inputs: Vec<f64>, targets: &[f64]) -> NllOutput {
        let batch_size = targets.len();
        let mut sum_e = vec![0.0; batch_size];
        let mut input_grads = vec![0.0; batch_size * Self::OUTPUT_SIZE];
        for b in 0..batch_size {
            let mut sum = 0.0;
            // sum of squared residuals
            let diff = inputs[b] - targets[b];
            sum += diff;
            // damping factor?
            input_grads[b] = diff / 1000.0;
            sum_e[b] = sum;
        }

        NllOutput {
            loss: sum_e,
            input_grads,
        }
    }

    /// return the RMSD of the outputs3 compared to the test labels
    fn check_output(&self, outputs3: Vec<f64>) -> f64 {
        let labels = self.test_labels();
        let mut sum = 0.0;
        let mut c = 0;
        for (l, o) in labels.iter().zip(outputs3) {
            let diff = l - o;
            sum += diff * diff;
            c += 1;
        }
        (sum / c as f64).sqrt()
    }
}
