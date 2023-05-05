#![allow(unused)]

pub struct Loss {
    last_inputs: Vec<f64>,
    fun: fn(f64) -> f64,
}

pub fn leaky_relu(x: f64) -> f64 {
    if x < 0.0 {
        0.01 * x
    } else {
        x
    }
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl Loss {
    pub fn new(fun: fn(f64) -> f64) -> Self {
        Self {
            fun,
            last_inputs: Vec::new(),
        }
    }

    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        self.last_inputs = inputs;
        self.last_inputs.iter().map(|&i| (self.fun)(i)).collect()
    }

    pub fn backward(&self, grads: Vec<f64>) -> Vec<f64> {
        let mut outputs = vec![0.0; grads.len()];
        for i in 0..self.last_inputs.len() {
            if self.last_inputs[i] < 0.0 {
                // NOTE zig code says grads[i] = but that doesn't make any sense
                outputs[i] = 0.01 * grads[i];
            } else {
                outputs[i] = grads[i]
            }
        }
        outputs
    }
}
