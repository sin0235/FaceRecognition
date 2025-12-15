# Vẽ histogram CONFIDENCE (Train / Val / Test)
import matplotlib.pyplot as plt
import numpy as np

def normalize_conf(conf):
    return 100 * (conf - conf.min()) / (conf.max() - conf.min() + 1e-6)

def plot_histogram(train_conf, val_conf, test_conf, threshold):
    plt.figure(figsize=(10,6))

    plt.hist(
        [normalize_conf(train_conf),
         normalize_conf(val_conf),
         normalize_conf(test_conf)],
        bins=30,
        label=["Train", "Val", "Test"],
        alpha=0.7
    )

    plt.axvline(
        normalize_conf(np.array([threshold]))[0],
        linestyle="--",
        label="Threshold"
    )

    plt.xlabel("Normalized Confidence (0–100)")
    plt.ylabel("Frequency")
    plt.title("LBPH Confidence Distribution")
    plt.legend()
    plt.show()
