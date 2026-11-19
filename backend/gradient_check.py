"""Compare analytical MLP gradients with finite-difference estimates."""

from __future__ import annotations

import numpy as np

from backend.mlp import MLP


def run_gradient_check(epsilon: float = 1e-5) -> dict[str, float]:
    """Check every parameter on a tiny network and return error statistics."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")

    rng = np.random.default_rng(2025)
    model = MLP(input_size=5, hidden1_size=4, hidden2_size=3, seed=11)
    features = rng.normal(size=5)
    target = 0.35
    model.b1[:] = 0.5
    model.b2[:] = 0.5
    analytical = model.loss_and_gradients(features, target)

    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    parameter_count = 0
    for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
        parameter = getattr(model, name)
        gradient = getattr(analytical, name)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            loss_plus = model.loss_and_gradients(features, target).loss
            parameter[index] = original - epsilon
            loss_minus = model.loss_and_gradients(features, target).loss
            parameter[index] = original

            numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
            absolute_error = abs(numerical - gradient[index])
            relative_error = absolute_error / max(
                1e-8, abs(numerical) + abs(gradient[index])
            )
            maximum_absolute_error = max(maximum_absolute_error, absolute_error)
            maximum_relative_error = max(maximum_relative_error, relative_error)
            parameter_count += 1

    return {
        "parameters_checked": float(parameter_count),
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
    }


if __name__ == "__main__":
    results = run_gradient_check()
    print(f"Checked {int(results['parameters_checked'])} parameters")
    print(f"Maximum absolute error: {results['maximum_absolute_error']:.3e}")
    print(f"Maximum relative error: {results['maximum_relative_error']:.3e}")
    if results["maximum_relative_error"] > 1e-4:
        raise SystemExit("Gradient check failed.")
    print("Gradient check passed.")
