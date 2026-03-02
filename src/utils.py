import numpy as np
from sklearn.metrics import confusion_matrix


def compute_cost(y_true, y_pred, fn_cost=5000, fp_cost=500):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * fp_cost + fn * fn_cost


def find_best_threshold_cost(y_true, y_prob, fn_cost=5000, fp_cost=500, steps=400):
    thresholds = np.linspace(0.01, 0.99, steps)
    best = {"threshold": 0.5, "cost": float("inf")}

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cost = compute_cost(y_true, y_pred, fn_cost=fn_cost, fp_cost=fp_cost)
        if cost < best["cost"]:
            best = {"threshold": float(t), "cost": float(cost)}

    return best