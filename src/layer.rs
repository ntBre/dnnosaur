use rand::{rngs::StdRng, Rng, SeedableRng};

pub struct LayerGrads {
    pub weight_grads: Vec<f64>,
    pub input_grads: Vec<f64>,
}

impl LayerGrads {
    fn new(weight_grads: Vec<f64>, input_grads: Vec<f64>) -> Self {
        Self {
            weight_grads,
            input_grads,
        }
    }
}

pub struct Layer {
    inputs: usize,
    outputs: usize,
    weights: Vec<f64>,
    last_inputs: Vec<f64>,
}

impl Layer {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        const SEED: u64 = 410;
        const SCALE: f64 = 0.2;
        let mut rng = StdRng::seed_from_u64(SEED);
        let mut weights = vec![0.0; inputs * outputs];
        weights.fill_with(|| SCALE * rng.gen_range(-1.0..=1.0));
        Self {
            weights,
            last_inputs: Vec::new(),
            inputs,
            outputs,
        }
    }

    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let batch_size = inputs.len() / self.inputs;
        let mut outputs = vec![0.0; batch_size * self.outputs];
        for b in 0..batch_size {
            for o in 0..self.outputs {
                let mut sum = 0.0;
                for i in 0..self.inputs {
                    sum += inputs[b * self.inputs + i]
                        * self.weights[self.outputs * i + o];
                }
                outputs[b * self.outputs + o] = sum;
            }
        }
        self.last_inputs = inputs;
        outputs
    }

    pub fn backward(&self, grads: Vec<f64>) -> LayerGrads {
        let mut weight_grads = vec![0.0; self.inputs * self.outputs];

        let batch_size = self.last_inputs.len() / self.inputs;
        let mut input_grads = vec![0.0; batch_size * self.inputs];

        for b in 0..batch_size {
            for i in 0..self.inputs {
                for o in 0..self.outputs {
                    weight_grads[i * self.outputs + o] += (grads
                        [b * self.outputs + o]
                        * self.last_inputs[b * self.inputs + i])
                        / batch_size as f64;
                    input_grads[b * self.inputs + i] += grads
                        [b * self.outputs + o]
                        * self.weights[i * self.outputs + o];
                }
            }
        }
        LayerGrads::new(weight_grads, input_grads)
    }

    pub fn apply_gradients(&mut self, grads: Vec<f64>) {
        for (i, w) in self.weights.iter_mut().enumerate() {
            const STEP_SIZE: f64 = 0.01;
            *w -= STEP_SIZE * grads[i];
        }
    }
}
