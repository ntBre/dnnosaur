pub struct Nll<const I: usize> {
    pub loss: Vec<f64>,
    pub input_grads: Vec<f64>,
}

impl<const I: usize> Nll<I> {
    pub fn nll(inputs: Vec<f64>, targets: Vec<u8>) -> Self {
        let batch_size = targets.len();
        let mut sum_e = vec![0.0; batch_size];
        let mut b = 0;
        while b < batch_size {
            let mut sum = 0.0;
            let mut i = 0;
            while i < I {
                sum += inputs[b * I + i].exp();
                i += 1;
            }
            sum_e[b] = sum;
            b += 1;
        }

        let mut loss = vec![0.0; batch_size];
        for b in 0..batch_size {
            loss[b] = -1.0
                * (inputs[b * I + targets[b] as usize].exp() / sum_e[b]).ln();
        }

        let mut input_grads = vec![0.0; batch_size * I];
        for b in 0..batch_size {
            for i in 0..I {
                input_grads[b * I + i] = inputs[b * I + i].exp() / sum_e[b];
                if i == targets[b] as usize {
                    input_grads[b * I + i] -= 1.0;
                }
            }
        }

        Self { loss, input_grads }
    }
}
