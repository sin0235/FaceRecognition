# TỰ ĐỘNG TÌM THRESHOLD TỐI ƯU (CORE)
import numpy as np
from .evaluate_lbph import evaluate_lbph

def find_best_threshold(model, val_faces, val_labels):
    best_thr = None
    best_score = -1

    for thr in range(40, 121, 5):
        acc, reject, _, used, coverage = evaluate_lbph(
            model, val_faces, val_labels, thr
        )

        score = acc - 0.5 * reject  # trade-off

        if score > best_score:
            best_score = score
            best_thr = thr

    return best_thr
