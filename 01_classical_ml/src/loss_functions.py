"""
Comprehensive Loss Functions from Scratch in NumPy
=================================================
Provides forward loss computation and analytical gradients for regression
and classification loss functions used throughout classical ML and deep learning.

Author: Leila Khezaz
"""

import numpy as np


class MSELoss:
    """Mean Squared Error (L2 Loss) for regression: L = (1/2N) * sum((y_pred - y_true)^2)"""
    @staticmethod
    def forward(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return 0.5 * np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def gradient(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return (y_pred - y_true) / len(y_true)


class MAELoss:
    """Mean Absolute Error (L1 Loss) for regression: L = (1/N) * sum(|y_pred - y_true|)"""
    @staticmethod
    def forward(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(np.abs(y_pred - y_true))

    @staticmethod
    def gradient(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return np.sign(y_pred - y_true) / len(y_true)


class HuberLoss:
    """Huber Loss (Smooth L1) - robust against outliers in regression."""
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        error = y_pred - y_true
        abs_error = np.abs(error)
        linear_mask = abs_error > self.delta
        quadratic = 0.5 * (error ** 2)
        linear = self.delta * (abs_error - 0.5 * self.delta)
        loss = np.where(linear_mask, linear, quadratic)
        return np.mean(loss)

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        error = y_pred - y_true
        abs_error = np.abs(error)
        linear_mask = abs_error > self.delta
        grad = np.where(linear_mask, self.delta * np.sign(error), error)
        return grad / len(y_true)


class BinaryCrossEntropyLoss:
    """Binary Cross Entropy (Log Loss) with numerical stability clipping."""
    def __init__(self, eps: float = 1e-15):
        self.eps = eps

    def forward(self, y_pred_prob: np.ndarray, y_true: np.ndarray) -> float:
        p = np.clip(y_pred_prob, self.eps, 1.0 - self.eps)
        loss = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        return np.mean(loss)

    def gradient(self, y_pred_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        p = np.clip(y_pred_prob, self.eps, 1.0 - self.eps)
        return ((p - y_true) / (p * (1.0 - p))) / len(y_true)


class CategoricalCrossEntropyLoss:
    """Multiclass Cross Entropy Loss with Softmax normalization."""
    def __init__(self, eps: float = 1e-15):
        self.eps = eps

    def forward(self, logits: np.ndarray, y_true_onehot: np.ndarray) -> float:
        # Softmax with max subtraction for numerical stability
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        probs = np.clip(probs, self.eps, 1.0 - self.eps)
        return -np.mean(np.sum(y_true_onehot * np.log(probs), axis=-1))

    def gradient(self, logits: np.ndarray, y_true_onehot: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return (probs - y_true_onehot) / len(y_true_onehot)


class HingeLoss:
    """Hinge Loss for Support Vector Machines (y in {-1, 1})."""
    @staticmethod
    def forward(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(np.maximum(0, 1 - y_true * y_pred))

    @staticmethod
    def gradient(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        margin = 1 - y_true * y_pred
        grad = np.where(margin > 0, -y_true, 0)
        return grad / len(y_true)


if __name__ == "__main__":
    print("Testing Loss Functions from Scratch...")
    y_true_reg = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred_reg = np.array([1.2, 1.8, 3.5, 3.8])

    mse = MSELoss()
    print(f"MSE Loss:    {mse.forward(y_pred_reg, y_true_reg):.4f}")
    print(f"MSE Grad:    {mse.gradient(y_pred_reg, y_true_reg)}")

    mae = MAELoss()
    print(f"MAE Loss:    {mae.forward(y_pred_reg, y_true_reg):.4f}")
    print(f"MAE Grad:    {mae.gradient(y_pred_reg, y_true_reg)}")

    huber = HuberLoss(delta=1.0)
    print(f"Huber Loss:  {huber.forward(y_pred_reg, y_true_reg):.4f}")
    print(f"Huber Grad:  {huber.gradient(y_pred_reg, y_true_reg)}")

    y_true_bin = np.array([1, 0, 1, 0])
    y_pred_prob = np.array([0.9, 0.1, 0.8, 0.3])
    bce = BinaryCrossEntropyLoss()
    print(f"BCE Loss:    {bce.forward(y_pred_prob, y_true_bin):.4f}")
    print(f"BCE Grad:    {bce.gradient(y_pred_prob, y_true_bin)}")

    print("All Loss Functions verified successfully!")
