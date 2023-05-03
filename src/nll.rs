pub struct Nll {
    pub n: usize,
    pub loss: Vec<f64>,
    pub input_grads: Vec<f64>,
}

impl Nll {
    pub fn new(n: usize, inputs: Vec<f64>, targets: &[u8]) -> Self {
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

        Self {
            n,
            loss,
            input_grads,
        }
    }
}
