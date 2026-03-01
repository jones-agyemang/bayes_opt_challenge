from sklearn import svm

from utils.loader import (
    load_input_data,
    load_output_data
)

from log_eval_plts import ( signed_log_plot )

func_id = 1
X = load_input_data(func_id)
y = load_output_data(func_id)

y_min, y_max = y.min(), y.max()

# breakpoint()
threshold = ( y_max - y_min ) / 2.
dist_from_mid = [ output - threshold for output in y ]

y_out = [ 1 if output > threshold else 0 for output in y]

len_x, len_y = len(X), len(y)
assert len_x == len_y, f"Length of X: {len_x} is unequal to length of y: {len_y}"

print(f"y(min): {y_min}")
print(f"y(max): {y_max}")
print(f"Thr: {threshold}")
print(y_max)
print(y_out)
print(dist_from_mid)

breakpoint()

# clf = svm.SVC(kernel="linear").fit(X, y)

# signed_log_plot(1, y, n_expensive=3)