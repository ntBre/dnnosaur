use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

use crate::NllOutput;
use crate::Train;

pub struct Qff {
    pub train_data: Vec<f64>,
    pub train_labels: Vec<f64>,
    pub test_data: Vec<f64>,
    pub test_labels: Vec<f64>,
}

impl Qff {
    /// load a [Qff] from the file specified by `p`. This file should contain
    /// the frequencies on the first line, followed by the lxm matrix
    pub fn load(dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let train = [
            //
            "benzene",
            "naphthalene",
            "phenanthrene",
        ];
        let (train_labels, train_data) = Self::load_files(
            train.iter().map(|t| dir.as_ref().join(t)).collect(),
        )?;

        let test = [
            //
            "benzene",
            "naphthalene",
            "phenanthrene",
        ];
        let (test_labels, test_data) = Self::load_files(
            test.iter().map(|t| dir.as_ref().join(t)).collect(),
        )?;

        Ok(Self {
            train_data,
            train_labels,
            test_data,
            test_labels,
        })
    }

    fn load_files(
        files: Vec<impl AsRef<Path> + std::fmt::Debug>,
    ) -> Result<(Vec<f64>, Vec<f64>), std::io::Error> {
        let mut freqs = Vec::new();
        let mut lxm = Vec::new();
        for f in files {
            let (f, l) = Self::load_one(f)?;
            freqs.extend(f);
            lxm.extend(l);
        }
        Ok((freqs, lxm))
    }

    fn load_one(
        p: impl AsRef<Path>,
    ) -> Result<(Vec<f64>, Vec<f64>), std::io::Error> {
        let f = File::open(p)?;
        let mut lines = BufReader::new(f).lines().flatten();
        let mut freqs: Vec<f64> = lines
            .next()
            .expect("no lines found")
            .split_ascii_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        freqs.resize_with(Self::LABEL_SIZE, || 0.0);
        let mut lxm: Vec<f64> = Vec::new();
        let r = (Self::INPUT_SIZE as f64).sqrt() as usize;
        for line in lines {
            let mut l: Vec<f64> = line
                .split_ascii_whitespace()
                .map(|s| s.parse::<f64>().unwrap())
                .collect();
            // skip blank lines
            if l.is_empty() {
                continue;
            }
            l.resize_with(r, || 0.0);
            lxm.extend(l);
        }
        // already padded each row, so if there are not enough rows, add zero
        // rows until the difference is made up. TODO do this in one resize call
        while lxm.len() < Self::INPUT_SIZE {
            lxm.extend(vec![0.0; r]);
        }
        Ok((freqs, lxm))
    }
}

impl Train<f64> for Qff {
    /// the maximum size of the input lxm matrices
    const INPUT_SIZE: usize = 4356;

    /// the maximum size of the output frequencies - 30 again for benzene
    const LABEL_SIZE: usize = 66;

    /// also the maximum size of the output frequencies - 30 for benzene alone
    const OUTPUT_SIZE: usize = 66;

    const BATCH_SIZE: usize = 1;

    const DATA_SIZE: usize = 3;

    fn train_data(&self, r: std::ops::Range<usize>) -> &[f64] {
        &self.train_data[r]
    }

    fn train_labels(&self, r: std::ops::Range<usize>) -> &[f64] {
        &self.train_labels[r]
    }

    fn test_data(&self) -> &[f64] {
        &self.test_data
    }

    fn test_labels(&self) -> &[f64] {
        &self.test_labels
    }

    fn nll(inputs: Vec<f64>, targets: &[f64]) -> NllOutput {
        let batch_size = targets.len();
        let mut loss = vec![0.0; batch_size];
        let mut input_grads = vec![0.0; batch_size * Self::OUTPUT_SIZE];
        for b in 0..batch_size {
            let diff = inputs[b] - targets[b];
            loss[b] = diff.abs();
            // damping factor?
            input_grads[b] = diff / 1000.0;
        }

        NllOutput { loss, input_grads }
    }

    /// return the RMSD of the outputs3 compared to the test labels
    fn check_output(&self, got: &[f64], want: &[f64]) -> f64 {
        let mut sum = 0.0;
        let mut c = 0;
        for (l, o) in want.iter().zip(got) {
            let diff = l - o;
            sum += diff * diff;
            c += 1;
        }
        (sum / c as f64).sqrt()
    }
}
