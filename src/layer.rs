use rand::Rng;

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

pub struct Layer<const I: usize, const O: usize>
where
    [f64; I * O]: Sized,
{
    weights: [f64; I * O],
    last_inputs: Vec<f64>,
}

impl<const I: usize, const O: usize> Layer<I, O>
where
    [f64; I * O]: Sized,
{
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        const SCALE: f64 = 0.2;
        Self {
            weights: core::array::from_fn(|_| {
                SCALE * rng.gen_range(-1.0..=1.0)
            }),
            last_inputs: Vec::new(),
        }
    }

    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let batch_size = inputs.len() / I;
        let mut outputs = vec![0.0; batch_size * O];
        for b in 0..batch_size {
            for o in 0..O {
                let mut sum = 0.0;
                for i in 0..I {
                    sum += inputs[b * I + i] * self.weights[O * i + o];
                }
                outputs[b * O + o] = sum;
            }
        }
        self.last_inputs = inputs;
        outputs
    }

    pub fn backward(&self, grads: Vec<f64>) -> LayerGrads {
        let mut weight_grads = vec![0.0; I * O];

        let batch_size = self.last_inputs.len() / I;
        let mut input_grads = vec![0.0; batch_size * I];

        for b in 0..batch_size {
            for i in 0..I {
                for o in 0..O {
                    weight_grads[i * O + o] += (grads[b * O + o]
                        * self.last_inputs[b * I + i])
                        / batch_size as f64;
                    input_grads[b * I + i] +=
                        grads[b * O + o] * self.weights[i * O + o];
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
