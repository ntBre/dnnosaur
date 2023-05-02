use std::io::{Read, Seek};

use crate::INPUT_SIZE;

#[derive(Debug)]
pub struct Data {
    train_images: Vec<f64>,
    train_labels: Vec<u8>,
    test_images: Vec<f64>,
    test_labels: Vec<u8>,
}

impl Data {
    pub fn read_mnist() -> Self {
        let bytes = Self::read_idx_file("data/train-images-idx3-ubyte", 16);
        let mut train_images = Vec::new();
        for i in 0..INPUT_SIZE * 60_000 {
            train_images.push(bytes[i] as f64 / 255.0);
        }

        let train_labels =
            Self::read_idx_file("data/train-labels-idx1-ubyte", 8);

        let bytes = Self::read_idx_file("data/t10k-images-idx3-ubyte", 16);
        let mut test_images = Vec::new();
        for i in 0..INPUT_SIZE * 10_000 {
            test_images.push(bytes[i] as f64 / 255.0);
        }

        let test_labels = Self::read_idx_file("data/t10k-labels-idx1-ubyte", 8);

        Self {
            train_images,
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
        return buf;
    }
}
