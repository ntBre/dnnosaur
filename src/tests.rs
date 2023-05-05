use approx::assert_abs_diff_eq;

use super::*;

#[test]
fn test_train() {
    let got = mnist::Data::default()
        .read_mnist()
        .train(3, LossFn::LeakyRelu);
    let want = vec![88.88, 90.94, 92.07];
    assert_abs_diff_eq!(got.as_slice(), want.as_slice(), epsilon = 1e-2);
}
