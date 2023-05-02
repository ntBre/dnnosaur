pub struct Nll<const I: usize> {
    loss: Vec<f64>,
    input_grads: Vec<f64>,
}

impl<const I: usize> Nll<I> {
    pub fn new(inputs: Vec<f64>, targets: Vec<usize>) -> Self {
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
            loss[b] = -1.0 * (inputs[b * I + targets[b]].exp() / sum_e[b]).ln();
        }

        let mut input_grads = vec![0.0; batch_size * I];
        for b in 0..batch_size {
            for i in 0..I {
                input_grads[b * I + i] = inputs[b * I + i].exp() / sum_e[b];
                if i == targets[b] {
                    input_grads[b * I + i] -= 1.0;
                }
            }
        }

        Self { loss, input_grads }
    }
}
