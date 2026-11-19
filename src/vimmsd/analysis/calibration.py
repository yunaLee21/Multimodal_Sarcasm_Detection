"""Hiệu chỉnh quyết định hậu kỳ bằng hệ số nhân xác suất từng lớp."""
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import f1_score


def apply_class_multipliers(probabilities, multipliers, normalize=True):
    """Tính alpha_c * P(y=c|x); lớp đầu là lớp tham chiếu khi tối ưu."""
    p = np.asarray(probabilities, dtype=np.float64)
    alpha = np.asarray(multipliers, dtype=np.float64)
    if p.ndim != 2 or alpha.shape != (p.shape[1],):
        raise ValueError("probabilities và multipliers có kích thước không tương thích")
    if not np.isfinite(p).all() or not np.isfinite(alpha).all() or (p < 0).any() or (alpha <= 0).any():
        raise ValueError("xác suất phải hữu hạn, không âm; hệ số nhân phải hữu hạn, dương")
    adjusted = p * alpha
    if normalize:
        adjusted = adjusted / adjusted.sum(axis=1, keepdims=True).clip(min=1e-12)
    return adjusted


def tune_class_multipliers(probabilities, labels, bounds=(0.1, 10.0), seed=42, maxiter=80, popsize=12):
    """Tối ưu macro-F1 trên tập calibration; giữ alpha_0=1 để loại bỏ bất biến tỉ lệ."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if p.ndim != 2 or p.shape[1] < 2 or y.shape != (p.shape[0],):
        raise ValueError("cần probabilities (N, C>=2) và labels (N,)")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("mỗi hàng probabilities phải hữu hạn, không âm và có tổng bằng 1")
    if not (0 < bounds[0] <= 1 <= bounds[1]):
        raise ValueError("bounds phải dương và chứa giá trị 1")

    baseline = f1_score(y, p.argmax(axis=1), average="macro", zero_division=0)

    def objective(log_alpha):
        alpha = np.exp(np.concatenate([[0.0], log_alpha]))
        preds = (p * alpha).argmax(axis=1)
        return -f1_score(y, preds, average="macro", zero_division=0)

    log_bounds = [(np.log(bounds[0]), np.log(bounds[1]))] * (p.shape[1] - 1)
    result = differential_evolution(
        objective, log_bounds, x0=np.zeros(p.shape[1] - 1), seed=seed, maxiter=maxiter,
        popsize=popsize, tol=1e-7, polish=False, updating="immediate", workers=1,
    )
    multipliers = np.exp(np.concatenate([[0.0], result.x]))
    diagnostics = {
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "baseline_macro_f1": float(baseline),
        "calibration_macro_f1": float(-result.fun),
    }
    return multipliers, diagnostics
