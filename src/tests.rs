use approx::assert_abs_diff_eq;

use super::*;

#[test]
fn test_train() {
    let got = train(mnist::Data::read_mnist());
    let want = vec![
        88.88, 90.94, 92.07, 92.79, 93.32, 93.74, 94.01, 94.38, 94.69, 94.88,
        95.04, 95.20, 95.40, 95.63, 95.81, 95.80, 95.88, 96.08, 96.12, 96.22,
        96.29, 96.35, 96.44, 96.51, 96.58,
    ];
    assert_abs_diff_eq!(got.as_slice(), want.as_slice(), epsilon = 1e-2);
}
