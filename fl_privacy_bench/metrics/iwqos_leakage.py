import numpy as np


def estimate_mi_mmse(x: np.ndarray, z: np.ndarray) -> float:
    """
    MMSE-based lower-bound leakage estimator used by IWQoS paper style:
      I_hat(X;Z) = h(X) - h(X - E[X|Z])
    with a linear MMSE approximation for E[X|Z].
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    n = min(x.size, z.size)
    if n < 2:
        return 0.0
    x = x[:n]
    z = z[:n]

    z_mean = np.mean(z)
    x_mean = np.mean(x)
    z_centered = z - z_mean
    x_centered = x - x_mean

    var_z = np.var(z_centered)
    if var_z <= 1e-18:
        return 0.0

    cov_xz = np.mean(x_centered * z_centered)
    a = cov_xz / (var_z + 1e-18)
    b = x_mean - a * z_mean
    x_hat = a * z + b
    residual = x - x_hat

    var_x = np.var(x)
    var_res = np.var(residual)
    if var_x <= 1e-18 or var_res <= 1e-18:
        return 0.0

    leakage = 0.5 * np.log(var_x / var_res)
    return float(max(0.0, leakage))
