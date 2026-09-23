import numpy as np

from cross_attention_interventions.metrics import best_threshold_iou, binary_iou


def test_binary_iou():
    a = np.array([[1, 0], [1, 0]], dtype=bool)
    b = np.array([[1, 0], [0, 1]], dtype=bool)
    assert abs(binary_iou(a, b) - 1 / 3) < 1e-9


def test_best_threshold_can_recover_target():
    h = np.array([[0.9, 0.1], [0.8, 0.2]])
    target = np.array([[1, 0], [1, 0]], dtype=bool)
    assert best_threshold_iou(h, target) == 1.0
