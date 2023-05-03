use std::io::{Read, Seek};

use crate::Train;

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

impl Train for Data {
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

    const INPUT_SIZE: usize = 784;
    const OUTPUT_SIZE: usize = 10;
    const BATCH_SIZE: usize = 32;
}
