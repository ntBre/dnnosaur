pub struct Relu {
    last_inputs: Vec<f64>,
}

impl Relu {
    pub fn new() -> Self {
        Self {
            last_inputs: Vec::new(),
        }
    }

    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        let mut outputs = vec![0.0; inputs.len()];
        let mut i = 0;
        while i < inputs.len() {
            if inputs[i] < 0.0 {
                outputs[i] = 0.01 * inputs[i];
            } else {
                outputs[i] = inputs[i];
            }
            i += 1;
        }
        self.last_inputs = inputs;
        outputs
    }

    pub fn backward(&self, grads: Vec<f64>) -> Vec<f64> {
	let mut outputs = vec![0.0; grads.len()];
	let mut i = 0;
	while i < self.last_inputs.len() {
	    if self.last_inputs[i] < 0.0 {
		// NOTE zig code says grads[i] = but that doesn't make any sense
		outputs[i] = 0.01 * grads[i];
	    } else {
		outputs[i] = grads[i]
	    }
	    i += 1;
	}
	outputs
    }
}
