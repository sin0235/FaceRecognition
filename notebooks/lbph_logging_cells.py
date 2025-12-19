# ==============================================================================
# CELL: Lưu Threshold Tuning Results
# ==============================================================================
# Thêm cell này sau cell threshold tuning
"""
# Lưu threshold tuning results
import pandas as pd

# Tạo plots directory
PLOTS_DIR = os.path.join(CHECKPOINT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Lưu threshold results vào JSON
threshold_history = []
for thr, acc, cov, score in threshold_results:
    threshold_history.append({
        "threshold": int(thr),
        "accuracy": float(acc),
        "coverage": float(cov),
        "score": float(score)
    })

THRESHOLD_HISTORY_PATH = os.path.join(CHECKPOINT_DIR, "threshold_history.json")
with open(THRESHOLD_HISTORY_PATH, "w") as f:
    json.dump(threshold_history, f, indent=2)

print(f"[OK] Threshold history saved: {THRESHOLD_HISTORY_PATH}")

# Lưu threshold results vào CSV
df_threshold = pd.DataFrame(threshold_history)
THRESHOLD_CSV_PATH = os.path.join(CHECKPOINT_DIR, "threshold_history.csv")
df_threshold.to_csv(THRESHOLD_CSV_PATH, index=False)
print(f"[OK] Threshold history CSV saved: {THRESHOLD_CSV_PATH}")
"""

# ==============================================================================
# CELL: Vẽ Biểu Đồ Threshold Tuning
# ==============================================================================
"""
# Vẽ biểu đồ threshold vs metrics
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

thresholds = [r[0] for r in threshold_results]
accuracies = [r[1] for r in threshold_results]
coverages = [r[2] for r in threshold_results]
scores = [r[3] for r in threshold_results]

# Plot 1: Threshold vs Accuracy
axes[0].plot(thresholds, accuracies, marker='o', linewidth=2, markersize=6, color='#2ecc71')
axes[0].axvline(best_threshold, color='red', linestyle='--', linewidth=2, label=f'Best: {best_threshold}')
axes[0].set_xlabel('Threshold', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Threshold vs Accuracy', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)

# Plot 2: Threshold vs Coverage
axes[1].plot(thresholds, coverages, marker='s', linewidth=2, markersize=6, color='#3498db')
axes[1].axvline(best_threshold, color='red', linestyle='--', linewidth=2, label=f'Best: {best_threshold}')
axes[1].set_xlabel('Threshold', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Coverage', fontsize=12, fontweight='bold')
axes[1].set_title('Threshold vs Coverage', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

# Plot 3: Threshold vs Combined Score
axes[2].plot(thresholds, scores, marker='^', linewidth=2, markersize=6, color='#e74c3c')
axes[2].axvline(best_threshold, color='red', linestyle='--', linewidth=2, label=f'Best: {best_threshold}')
axes[2].set_xlabel('Threshold', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Combined Score', fontsize=12, fontweight='bold')
axes[2].set_title('Threshold vs Combined Score', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].legend(fontsize=10)

plt.tight_layout()

# Save plot
THRESHOLD_PLOT_PATH = os.path.join(PLOTS_DIR, "threshold_tuning.png")
plt.savefig(THRESHOLD_PLOT_PATH, dpi=150, bbox_inches='tight')
print(f"[OK] Threshold plot saved: {THRESHOLD_PLOT_PATH}")

plt.show()
"""

# ==============================================================================
# CELL: Vẽ Confidence Distribution (Validation Set)
# ==============================================================================
"""
# Vẽ confidence distribution trên validation set
print("="*60)
print("ANALYZING CONFIDENCE DISTRIBUTION (VALIDATION SET)")
print("="*60)

# Thu thập tất cả predictions và confidences
val_predictions = []
val_confidences_all = []
val_true_labels = []

for i, (face, true_label) in enumerate(zip(val_faces, val_labels)):
    pred_label, confidence = model.predict(face)
    val_predictions.append(pred_label)
    val_confidences_all.append(confidence)
    val_true_labels.append(true_label)

val_confidences_all = np.array(val_confidences_all)
val_predictions = np.array(val_predictions)

# Phân loại correct và incorrect predictions
correct_mask = val_predictions == val_true_labels
correct_confidences = val_confidences_all[correct_mask]
incorrect_confidences = val_confidences_all[~correct_mask]

print(f"Total predictions: {len(val_confidences_all)}")
print(f"Correct: {len(correct_confidences)} ({len(correct_confidences)/len(val_confidences_all)*100:.1f}%)")
print(f"Incorrect: {len(incorrect_confidences)} ({len(incorrect_confidences)/len(val_confidences_all)*100:.1f}%)")

# Vẽ distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot 1: Histogram of all confidences
axes[0].hist(correct_confidences, bins=50, alpha=0.7, color='green', label='Correct', edgecolor='black')
axes[0].hist(incorrect_confidences, bins=50, alpha=0.7, color='red', label='Incorrect', edgecolor='black')
axes[0].axvline(best_threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold: {best_threshold}')
axes[0].set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title('Confidence Distribution (Val Set)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Plot 2: Box plot
data_to_plot = [correct_confidences, incorrect_confidences]
box = axes[1].boxplot(data_to_plot, labels=['Correct', 'Incorrect'], patch_artist=True)
box['boxes'][0].set_facecolor('lightgreen')
box['boxes'][1].set_facecolor('lightcoral')
axes[1].axhline(best_threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold: {best_threshold}')
axes[1].set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
axes[1].set_title('Confidence Box Plot (Val Set)', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save plot
CONFIDENCE_PLOT_PATH = os.path.join(PLOTS_DIR, "confidence_distribution_val.png")
plt.savefig(CONFIDENCE_PLOT_PATH, dpi=150, bbox_inches='tight')
print(f"[OK] Confidence plot saved: {CONFIDENCE_PLOT_PATH}")

plt.show()

# Lưu confidence statistics
confidence_stats = {
    "correct": {
        "mean": float(np.mean(correct_confidences)),
        "std": float(np.std(correct_confidences)),
        "min": float(np.min(correct_confidences)),
        "max": float(np.max(correct_confidences)),
        "median": float(np.median(correct_confidences)),
        "count": int(len(correct_confidences))
    },
    "incorrect": {
        "mean": float(np.mean(incorrect_confidences)),
        "std": float(np.std(incorrect_confidences)),
        "min": float(np.min(incorrect_confidences)),
        "max": float(np.max(incorrect_confidences)),
        "median": float(np.median(incorrect_confidences)),
        "count": int(len(incorrect_confidences))
    },
    "threshold": int(best_threshold)
}

CONFIDENCE_STATS_PATH = os.path.join(CHECKPOINT_DIR, "confidence_stats.json")
with open(CONFIDENCE_STATS_PATH, "w") as f:
    json.dump(confidence_stats, f, indent=2)

print(f"[OK] Confidence stats saved: {CONFIDENCE_STATS_PATH}")
"""

# ==============================================================================
# CELL: Vẽ Confusion Matrix (Top N Classes)
# ==============================================================================
"""
# Vẽ confusion matrix cho top N classes có nhiều samples nhất
from sklearn.metrics import confusion_matrix
import numpy as np

print("="*60)
print("GENERATING CONFUSION MATRIX (VALIDATION SET)")
print("="*60)

# Chọn top N classes có nhiều samples nhất trong val set
from collections import Counter
label_counts = Counter(val_labels)
top_n = 10  # Hiển thị top 10 classes
top_classes = [label for label, count in label_counts.most_common(top_n)]

print(f"Top {top_n} classes: {top_classes}")

# Filter predictions và labels cho top classes
top_mask = np.isin(val_true_labels, top_classes)
top_true = np.array(val_true_labels)[top_mask]
top_pred = val_predictions[top_mask]

# KHÔNG FILTER theo threshold - hiển thị tất cả predictions
# Để đánh giá toàn diện khả năng phân loại của model
print(f"Vẽ confusion matrix cho {len(top_true)} predictions (không filter theo threshold)")

if len(top_true) > 0:
    # Tạo confusion matrix
    cm = confusion_matrix(top_true, top_pred, labels=top_classes)
    
    # Normalize confusion matrix (safe division)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.divide(cm.astype('float'), row_sums, 
                              where=row_sums!=0,
                              out=np.zeros_like(cm, dtype=float))
    
    # Vẽ confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=top_classes, yticklabels=top_classes,
                cbar_kws={'label': 'Normalized Frequency'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix (Top {top_n} Classes, All Predictions)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    CM_PLOT_PATH = os.path.join(PLOTS_DIR, "confusion_matrix_val.png")
    plt.savefig(CM_PLOT_PATH, dpi=150, bbox_inches='tight')
    print(f"[OK] Confusion matrix saved: {CM_PLOT_PATH}")
    
    plt.show()
else:
    print("[WARNING] Không có predictions nào cho top classes")
"""

# ==============================================================================
# CELL: Vẽ Test Set Confidence Distribution
# ==============================================================================
"""
# Vẽ confidence distribution cho test set
print("="*60)
print("ANALYZING CONFIDENCE DISTRIBUTION (TEST SET)")
print("="*60)

# Thu thập predictions và confidences cho test set
test_predictions = []
test_confidences_all = []
test_true_labels = test_labels.tolist()

for i, (face, true_label) in enumerate(zip(test_faces, test_labels)):
    pred_label, confidence = model.predict(face)
    test_predictions.append(pred_label)
    test_confidences_all.append(confidence)

test_confidences_all = np.array(test_confidences_all)
test_predictions = np.array(test_predictions)

# Phân loại correct và incorrect
test_correct_mask = test_predictions == test_labels
test_correct_confidences = test_confidences_all[test_correct_mask]
test_incorrect_confidences = test_confidences_all[~test_correct_mask]

print(f"Total predictions: {len(test_confidences_all)}")
print(f"Correct: {len(test_correct_confidences)} ({len(test_correct_confidences)/len(test_confidences_all)*100:.1f}%)")
print(f"Incorrect: {len(test_incorrect_confidences)} ({len(test_incorrect_confidences)/len(test_confidences_all)*100:.1f}%)")

# Vẽ histogram
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].hist(test_correct_confidences, bins=50, alpha=0.7, color='green', label='Correct', edgecolor='black')
axes[0].hist(test_incorrect_confidences, bins=50, alpha=0.7, color='red', label='Incorrect', edgecolor='black')
axes[0].axvline(best_threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold: {best_threshold}')
axes[0].set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title('Confidence Distribution (Test Set)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Box plot
data_to_plot = [test_correct_confidences, test_incorrect_confidences]
box = axes[1].boxplot(data_to_plot, labels=['Correct', 'Incorrect'], patch_artist=True)
box['boxes'][0].set_facecolor('lightgreen')
box['boxes'][1].set_facecolor('lightcoral')
axes[1].axhline(best_threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold: {best_threshold}')
axes[1].set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
axes[1].set_title('Confidence Box Plot (Test Set)', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save plot
TEST_CONFIDENCE_PLOT_PATH = os.path.join(PLOTS_DIR, "confidence_distribution_test.png")
plt.savefig(TEST_CONFIDENCE_PLOT_PATH, dpi=150, bbox_inches='tight')
print(f"[OK] Test confidence plot saved: {TEST_CONFIDENCE_PLOT_PATH}")

plt.show()

# Lưu test confidence stats
test_confidence_stats = {
    "correct": {
        "mean": float(np.mean(test_correct_confidences)),
        "std": float(np.std(test_correct_confidences)),
        "min": float(np.min(test_correct_confidences)),
        "max": float(np.max(test_correct_confidences)),
        "median": float(np.median(test_correct_confidences)),
        "count": int(len(test_correct_confidences))
    },
    "incorrect": {
        "mean": float(np.mean(test_incorrect_confidences)),
        "std": float(np.std(test_incorrect_confidences)),
        "min": float(np.min(test_incorrect_confidences)),
        "max": float(np.max(test_incorrect_confidences)),
        "median": float(np.median(test_incorrect_confidences)),
        "count": int(len(test_incorrect_confidences))
    },
    "threshold": int(best_threshold)
}

TEST_CONFIDENCE_STATS_PATH = os.path.join(CHECKPOINT_DIR, "test_confidence_stats.json")
with open(TEST_CONFIDENCE_STATS_PATH, "w") as f:
    json.dump(test_confidence_stats, f, indent=2)

print(f"[OK] Test confidence stats saved: {TEST_CONFIDENCE_STATS_PATH}")
"""

# ==============================================================================
# CELL: Cập nhật metadata với plots info
# ==============================================================================
"""
# Cập nhật metadata để bao gồm thông tin về plots
metadata["plots"] = {
    "threshold_tuning": "plots/threshold_tuning.png",
    "confidence_distribution_val": "plots/confidence_distribution_val.png",
    "confidence_distribution_test": "plots/confidence_distribution_test.png",
    "confusion_matrix_val": "plots/confusion_matrix_val.png"
}

metadata["confidence_stats_val"] = confidence_stats
metadata["confidence_stats_test"] = test_confidence_stats

# Lưu lại metadata đã cập nhật
with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"[OK] Metadata updated with plots info")
"""
