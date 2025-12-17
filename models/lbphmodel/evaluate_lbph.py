import numpy as np

def evaluate_lbph(model, faces, labels, threshold):
    correct = 0
    rejected = 0
    used = 0
    confidences = []

    for img, true_label in zip(faces, labels):
        pred, conf = model.predict(img)
        confidences.append(conf)

        if conf < threshold:
            used += 1
            if pred == true_label:
                correct += 1
        else:
            rejected += 1

    acc = (correct / used * 100) if used > 0 else 0
    reject_rate = rejected / len(labels) * 100
    coverage = used / len(labels) if len(labels) > 0 else 0

    return acc, reject_rate, np.array(confidences), used, coverage

