import numpy as np

def evaluate_lbph(model, faces, labels, threshold):
    correct = 0
    rejected = 0
    confidences = []

    for img, true_label in zip(faces, labels):
        pred, conf = model.predict(img)
        confidences.append(conf)

        if conf < threshold:
            if pred == true_label:
                correct += 1
        else:
            rejected += 1

    acc = correct / len(labels) * 100
    reject_rate = rejected / len(labels) * 100

    return acc, reject_rate, np.array(confidences)
