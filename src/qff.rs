use std::fmt::Debug;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

use crate::NllOutput;
use crate::Train;

#[derive(Default)]
pub struct Qff {
    pub train_data: Vec<f64>,
    pub train_labels: Vec<f64>,
    pub test_data: Vec<f64>,
    pub test_labels: Vec<f64>,

    input_size: usize,
    label_size: usize,
    output_size: usize,
    data_size: usize,
}

struct Load {
    freqs: Vec<f64>,
    lxm: Vec<f64>,
    max_freqs: usize,
    max_row: usize,
    max_col: usize,
    count: usize,
}

fn transpose(v: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut ret = vec![vec![0.0; v.len()]; v[0].len()];
    for i in 0..v.len() {
        (0..v[0].len()).for_each(|j| {
            ret[j][i] = v[i][j];
        });
    }
    ret
}

impl Qff {
    /// load a [Qff] from the file specified by `p`. This file should contain
    /// the frequencies on the first line, followed by the lxm matrix
    pub fn load(dir: impl AsRef<Path>) -> io::Result<Self> {
        Self::load_pahdb(dir)
    }

    pub fn load_pahdb(dir: impl AsRef<Path>) -> io::Result<Self> {
        // load all of the files from the pahdb
        let home = std::env::var("HOME").unwrap();
        let home = Path::new(&home);
        let data_dir = home.join("data/pahdb");
        let files: Vec<_> =
            data_dir.read_dir()?.flatten().map(|t| t.path()).collect();
        let files = &files[..20];
        // use 7/10 for training and 3/10 for validation
        let pivot = files.len() * 7 / 10;
        let train = &files[..pivot];
        let Load {
            freqs: train_labels,
            lxm: train_data,
            max_freqs,
            max_row,
            max_col,
            count,
        } = Self::load_files(
            train.iter().map(|t| dir.as_ref().join(t)).collect(),
        )?;

        let test = &files[pivot..];
        let Load {
            freqs: test_labels,
            lxm: test_data,
            ..
        } = Self::load_files(
            test.iter().map(|t| dir.as_ref().join(t)).collect(),
        )?;

        Ok(Self {
            train_data,
            train_labels,
            test_data,
            test_labels,
            input_size: max_row * max_col,
            label_size: max_freqs,
            output_size: max_freqs,
            data_size: count,
        })
    }

    pub fn load_local(&self, dir: impl AsRef<Path>) -> io::Result<Self> {
        let train = [
            //
            "benzene",
            "naphthalene",
            "phenanthrene",
        ];
        let Load {
            freqs: train_labels,
            lxm: train_data,
            max_freqs,
            max_row,
            max_col,
            count,
        } = Self::load_files(
            train.iter().map(|t| dir.as_ref().join(t)).collect(),
        )?;

        let test = [
            //
            "benzene",
            "naphthalene",
            "phenanthrene",
        ];
        let Load {
            freqs: test_labels,
            lxm: test_data,
            ..
        } = Self::load_files(
            test.iter().map(|t| dir.as_ref().join(t)).collect(),
        )?;

        Ok(Self {
            train_data,
            train_labels,
            test_data,
            test_labels,
            input_size: max_row * max_col,
            label_size: max_freqs,
            output_size: max_freqs,
            data_size: count,
        })
    }

    fn load_files(files: Vec<impl AsRef<Path> + Debug>) -> io::Result<Load> {
        let mut freqs = Vec::new();
        let mut lxm = Vec::new();
        let mut max_freqs = 0;
        let mut max_lxm = (0, 0);
        for f in files {
            let (f, l) = Self::load_one(f)?;
            max_freqs = f.len().max(max_freqs);
            let r = l.len();
            let c = match l.first() {
                Some(v) => v.len(),
                None => 0,
            };
            let (or, oc) = max_lxm;
            max_lxm = (or.max(r), oc.max(c));
            freqs.push(f);
            lxm.push(transpose(l));
        }
        // TODO do the padding here, after loading all of the files and tracking
        // the max for everything. do not do the padding in `load_one`
        for freq in freqs.iter_mut() {
            freq.resize(max_freqs, 0.0);
        }
        let (max_row, max_col) = max_lxm;
        for l in lxm.iter_mut() {
            for row in l.iter_mut() {
                row.resize(max_col, 0.0);
            }
            l.resize(max_row, vec![0.0; max_col]);
        }
        let count = freqs.len();
        assert_eq!(count, lxm.len());
        Ok(Load {
            freqs: freqs.into_iter().flatten().collect(),
            lxm: lxm.into_iter().flatten().flatten().collect(),
            max_freqs,
            max_row,
            max_col,
            count,
        })
    }

    fn load_one(
        p: impl AsRef<Path>,
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>), io::Error> {
        let f = File::open(p)?;
        let mut lines = BufReader::new(f).lines().flatten();
        let freqs: Vec<f64> = lines
            .next()
            .expect("no lines found")
            .split_ascii_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let mut lxm = Vec::new();
        for line in lines {
            let l: Vec<f64> = line
                .split_ascii_whitespace()
                .map(|s| s.parse::<f64>().unwrap())
                .collect();
            // skip blank lines
            if l.is_empty() {
                continue;
            }
            lxm.push(l);
        }
        Ok((freqs, lxm))
    }
}

impl Train<f64> for Qff {
    /// the maximum size of the input lxm matrices
    fn input_size(&self) -> usize {
        self.input_size
    }

    /// the maximum size of the output frequencies - 30 again for benzene
    fn label_size(&self) -> usize {
        self.label_size
    }

    /// also the maximum size of the output frequencies - 30 for benzene alone
    fn output_size(&self) -> usize {
        self.output_size
    }

    fn batch_size(&self) -> usize {
        2
    }

    fn data_size(&self) -> usize {
        self.data_size
    }

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

    fn nll(&self, inputs: Vec<f64>, targets: &[f64]) -> NllOutput {
        let batch_size = targets.len();
        let mut loss = vec![0.0; batch_size];
        let mut input_grads = vec![0.0; batch_size * self.output_size()];
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
