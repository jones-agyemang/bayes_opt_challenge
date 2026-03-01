import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

import tensorflow as tf
from tensorflow import keras


from utils.loader import (
    load_input_data,
    load_output_data
)

# -----------------------------
# 1) Turn BO regression data into "good" vs "bad"
# -----------------------------
def make_good_bad_labels(Y, method="quantile", q=0.80, tau=0.0):
    """
    Create binary labels from continuous BO outputs.

    method:
      - "quantile": good if y >= quantile(Y, q)
      - "threshold": good if y >= tau

    Returns: y_class (0/1), threshold_used
    """
    Y = np.asarray(Y).reshape(-1)

    if method == "quantile":
        threshold = float(np.quantile(Y, q))
    elif method == "threshold":
        threshold = float(tau)
    else:
        raise ValueError("method must be 'quantile' or 'threshold'")

    y_class = (Y >= threshold).astype(int)
    return y_class, threshold


# -----------------------------
# 2) Build a small NN classifier
# -----------------------------
def build_nn_classifier(input_dim, hidden=(32, 32), dropout=0.15, lr=1e-3):
    """
    Simple MLP binary classifier with dropout.
    Outputs P(good|x).
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dim,)))

    for h in hidden:
        model.add(keras.layers.Dense(h, activation="relu"))
        if dropout and dropout > 0:
            model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"]
    )
    return model


# -----------------------------
# 3) Train + Evaluate boundary approximation
# -----------------------------
def train_and_evaluate_nn_boundary(
    X, Y,
    label_method="quantile",
    q=0.80,
    tau=0.0,
    test_size=0.30,
    random_state=42,
    hidden=(32, 32),
    dropout=0.15,
    lr=1e-3,
    epochs=500,
    batch_size=16,
    patience=30,
):
    """
    Trains an NN classifier to learn the "good vs bad" boundary.
    Prints classification metrics + returns trained artifacts.
    """
    X = np.asarray(X)
    Y = np.asarray(Y).reshape(-1)

    y_class, threshold_used = make_good_bad_labels(Y, method=label_method, q=q, tau=tau)

    # Train/test split (stratify helps if classes are imbalanced)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=test_size, random_state=random_state, stratify=y_class
    )

    # Scale inputs (very important for NNs)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = build_nn_classifier(
        input_dim=X.shape[1],
        hidden=hidden,
        dropout=dropout,
        lr=lr
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train_s, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks
    )

    # Predictions
    p_test = model.predict(X_test_s, verbose=0).reshape(-1)
    y_pred = (p_test >= 0.5).astype(int)

    # Metrics (how well it approximated the boundary)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else float("nan")
    cm = confusion_matrix(y_test, y_pred)

    print("----- GOOD/BAD LABELING -----")
    print(f"Label method: {label_method}")
    print(f"Threshold used: {threshold_used:.6g}")
    print(f"Class balance (good=1): {y_class.mean():.3f}")

    print("\n----- TEST METRICS (Boundary Approximation) -----")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"ROC AUC:   {auc:.3f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    return {
        "model": model,
        "scaler": scaler,
        "threshold_used": threshold_used,
        "history": history,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "p_test": p_test, "y_pred": y_pred,
        "y_class_all": y_class,
    }


# -----------------------------
# 4) Visualise the learned decision boundary (2D only)
# -----------------------------
def plot_nn_decision_boundary_2d(X, y_class, model, scaler, title="NN Decision Boundary", grid_n=300):
    """
    Visualise P(good|x) across the 2D domain and overlay points.
    Only works when X has exactly 2 columns.
    """
    X = np.asarray(X)
    y_class = np.asarray(y_class).reshape(-1)

    if X.shape[1] != 2:
        raise ValueError("plot_nn_decision_boundary_2d only supports 2D inputs (X.shape[1] == 2).")

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    pad1 = 0.05 * (x1_max - x1_min + 1e-12)
    pad2 = 0.05 * (x2_max - x2_min + 1e-12)

    xs = np.linspace(x1_min - pad1, x1_max + pad1, grid_n)
    ys = np.linspace(x2_min - pad2, x2_max + pad2, grid_n)
    xx, yy = np.meshgrid(xs, ys)

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_s = scaler.transform(grid)
    probs = model.predict(grid_s, verbose=0).reshape(xx.shape)

    plt.figure()
    # Probability field
    plt.contourf(xx, yy, probs, levels=25)
    # 0.5 boundary
    plt.contour(xx, yy, probs, levels=[0.5])

    # Points
    good_mask = (y_class == 1)
    bad_mask = ~good_mask
    plt.scatter(X[bad_mask, 0], X[bad_mask, 1], marker="x", label="Bad (0)")
    plt.scatter(X[good_mask, 0], X[good_mask, 1], marker="o", label="Good (1)")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


# -----------------------------
# 5) Example usage with YOUR data (paste your X and Y)
# -----------------------------
# X = inputs  # shape (n, 2) or (n, d)
# Y = outputs # shape (n,)

# Example:
# results = train_and_evaluate_nn_boundary(
#     X, Y,
#     label_method="quantile", q=0.80,   # good = top 20% of observed outputs
#     # OR: label_method="threshold", tau=0.0  # good = y >= 0
#     hidden=(32, 32),
#     dropout=0.15,
#     lr=1e-3,
#     epochs=500,
#     batch_size=8,
#     patience=40
# )
#
# # Visualise boundary if 2D:
# y_class_all, thr = make_good_bad_labels(Y, method="quantile", q=0.80)
# plot_nn_decision_boundary_2d(
#     X, y_class_all,
#     results["model"], results["scaler"],
#     title=f"NN Boundary (good if y >= {thr:.3g})"
# )

X, Y = load_input_data(2), load_output_data(2)
results = train_and_evaluate_nn_boundary(
    X, Y,
    label_method="quantile", q=0.80,   # good = top 20% of observed outputs
    # OR: label_method="threshold", tau=0.0  # good = y >= 0
    hidden=(32, 32),
    dropout=0.15,
    lr=1e-3,
    epochs=500,
    batch_size=8,
    patience=40
)

# Visualise boundary if 2D:
y_class_all, thr = make_good_bad_labels(Y, method="quantile", q=0.80)
plot_nn_decision_boundary_2d(
    X, y_class_all,
    results["model"], results["scaler"],
    title=f"NN Boundary (good if y >= {thr:.3g})"
)