import json

facenet_cells = [
    {"cell_type": "markdown", "metadata": {}, "source": ["# Phan Tich Toan Dien FaceNet Model\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import json\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from pathlib import Path\n",
        "from IPython.display import display, Markdown, Image\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "plt.style.use('seaborn-v0_8-whitegrid')\n",
        "LOGS_DIR = Path('../logs/facenet')\n",
        "print(list(LOGS_DIR.glob('*')))\n"
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "with open(LOGS_DIR / 'training_history.json', 'r') as f:\n",
        "    training_data = json.load(f)\n",
        "with open(LOGS_DIR / 'facenet_evaluation_report.json', 'r') as f:\n",
        "    eval_report = json.load(f)\n",
        "metrics = eval_report['metrics']\n",
        "print(f'Epochs: {len(training_data[\"train_loss\"])}')\n"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": ["## Training Curves\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
        "epochs = list(range(1, len(training_data['train_loss']) + 1))\n",
        "axes[0,0].plot(epochs, training_data['train_loss'], 'b-', lw=2, label='Train')\n",
        "axes[0,0].plot(epochs, training_data['val_loss'], 'r-', lw=2, label='Val')\n",
        "axes[0,0].set_title('Triplet Loss'); axes[0,0].legend(); axes[0,0].grid(True)\n",
        "train_acc = [x*100 for x in training_data['train_acc']]\n",
        "val_acc = [x*100 for x in training_data['val_acc']]\n",
        "axes[0,1].plot(epochs, train_acc, 'b-', lw=2, label='Train')\n",
        "axes[0,1].plot(epochs, val_acc, 'r-', lw=2, label='Val')\n",
        "axes[0,1].set_title('Accuracy'); axes[0,1].legend(); axes[0,1].grid(True)\n",
        "gap = np.array(train_acc) - np.array(val_acc)\n",
        "axes[1,0].fill_between(epochs, 0, gap, alpha=0.4, color='orange')\n",
        "axes[1,0].set_title('Generalization Gap'); axes[1,0].grid(True)\n",
        "axes[1,1].plot(epochs, training_data['lr'], 'g-', lw=2)\n",
        "axes[1,1].set_yscale('log'); axes[1,1].set_title('Learning Rate'); axes[1,1].grid(True)\n",
        "plt.tight_layout()\n",
        "plt.savefig(LOGS_DIR / 'training_analysis.png', dpi=150)\n",
        "plt.show()\n"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": ["## Evaluation Metrics\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "print('='*60)\n",
        "print('FACENET EVALUATION SUMMARY')\n",
        "print('='*60)\n",
        "print(f'Top-1 Accuracy: {metrics[\"top1_accuracy\"]:.2f}%')\n",
        "print(f'Top-5 Accuracy: {metrics[\"top5_accuracy\"]:.2f}%')\n",
        "print(f'AUC-ROC: {metrics[\"auc\"]:.4f}')\n",
        "print(f'EER: {metrics[\"eer\"]*100:.2f}%')\n",
        "print(f'Embedding Size: {eval_report[\"embedding_size\"]}')\n"
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "fig, ax = plt.subplots(figsize=(10, 5))\n",
        "names = ['Top-1', 'Top-5', 'AUC*100']\n",
        "vals = [metrics['top1_accuracy'], metrics['top5_accuracy'], metrics['auc']*100]\n",
        "ax.bar(names, vals, color=['#e74c3c', '#3498db', '#9b59b6'])\n",
        "ax.set_ylim([75, 100])\n",
        "ax.set_title('FaceNet Metrics')\n",
        "for i, v in enumerate(vals): ax.text(i, v+0.5, f'{v:.1f}', ha='center')\n",
        "plt.tight_layout()\n",
        "plt.savefig(LOGS_DIR / 'metrics_visualization.png', dpi=150)\n",
        "plt.show()\n"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": ["## Existing Visualizations\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "for img in ['facenet_confusion_matrix.png', 'facenet_roc_curve.png', 'facenet_threshold_analysis.png']:\n",
        "    p = LOGS_DIR / img\n",
        "    if p.exists(): display(Image(filename=str(p), width=700))\n"
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "summary = {'model': 'FaceNet', 'embedding_size': eval_report['embedding_size'],\n",
        "           'epochs': len(training_data['train_loss']),\n",
        "           'top1': metrics['top1_accuracy'], 'top5': metrics['top5_accuracy'],\n",
        "           'auc': metrics['auc'], 'eer': metrics['eer']}\n",
        "with open(LOGS_DIR / 'comprehensive_summary.json', 'w') as f: json.dump(summary, f, indent=2)\n",
        "print(json.dumps(summary, indent=2))\n"
    ]}
]

lbph_cells = [
    {"cell_type": "markdown", "metadata": {}, "source": ["# Phan Tich Toan Dien LBPH Model\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import json\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from pathlib import Path\n",
        "from IPython.display import display, Image\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "plt.style.use('seaborn-v0_8-whitegrid')\n",
        "LOGS_DIR = Path('../logs/LBHP')\n",
        "print(list(LOGS_DIR.glob('*')))\n"
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "with open(LOGS_DIR / 'metadata.json', 'r') as f: metadata = json.load(f)\n",
        "with open(LOGS_DIR / 'confidence_stats.json', 'r') as f: conf_stats = json.load(f)\n",
        "with open(LOGS_DIR / 'threshold_history.json', 'r') as f: threshold_history = json.load(f)\n",
        "print(f'Num classes: {metadata[\"num_classes\"]}, Test acc: {metadata[\"test_accuracy\"]*100:.2f}%')\n"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": ["## Model Configuration\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "print('='*60)\n",
        "print('LBPH MODEL CONFIGURATION')\n",
        "print('='*60)\n",
        "params = metadata['model_params']\n",
        "print(f'Radius: {params[\"radius\"]}, Neighbors: {params[\"neighbors\"]}')\n",
        "print(f'Grid: {params[\"grid_x\"]}x{params[\"grid_y\"]}')\n",
        "print(f'Num Classes: {metadata[\"num_classes\"]}')\n",
        "print(f'Threshold: {metadata[\"threshold\"]}')\n",
        "print(f'Test Accuracy: {metadata[\"test_accuracy\"]*100:.2f}%')\n",
        "print(f'Test Coverage: {metadata[\"test_coverage\"]*100:.2f}%')\n"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": ["## Threshold Analysis\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "th_df = pd.DataFrame(threshold_history)\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "axes[0].plot(th_df['threshold'], th_df['accuracy']*100, 'b-o', lw=2)\n",
        "axes[0].set_xlabel('Threshold'); axes[0].set_ylabel('Accuracy (%)')\n",
        "axes[0].set_title('Accuracy vs Threshold'); axes[0].grid(True)\n",
        "ax2 = axes[1]\n",
        "ax2.plot(th_df['threshold'], th_df['accuracy']*100, 'b-o', lw=2, label='Accuracy')\n",
        "ax2_twin = ax2.twinx()\n",
        "ax2_twin.plot(th_df['threshold'], th_df['coverage']*100, 'g-s', lw=2, label='Coverage')\n",
        "ax2.set_xlabel('Threshold'); ax2.set_ylabel('Accuracy (%)', color='b')\n",
        "ax2_twin.set_ylabel('Coverage (%)', color='g')\n",
        "ax2.set_title('Acc vs Coverage Trade-off'); ax2.grid(True)\n",
        "plt.tight_layout()\n",
        "plt.savefig(LOGS_DIR / 'threshold_analysis_comprehensive.png', dpi=150)\n",
        "plt.show()\n",
        "display(th_df)\n"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": ["## Confidence Distribution\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "print('Correct:', conf_stats['correct'])\n",
        "print('Incorrect:', conf_stats['incorrect'])\n"
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "fig, ax = plt.subplots(figsize=(10, 6))\n",
        "cm, cs = conf_stats['correct']['mean'], conf_stats['correct']['std']\n",
        "im, ist = conf_stats['incorrect']['mean'], conf_stats['incorrect']['std']\n",
        "x = np.linspace(0, 150, 500)\n",
        "yc = (1/(cs*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-cm)/cs)**2)\n",
        "yi = (1/(ist*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-im)/ist)**2)\n",
        "ax.fill_between(x, yc, alpha=0.5, color='green', label=f'Correct (mean={cm:.1f})')\n",
        "ax.fill_between(x, yi, alpha=0.5, color='red', label=f'Incorrect (mean={im:.1f})')\n",
        "ax.axvline(x=metadata['threshold'], color='black', linestyle='--', lw=2, label='Threshold')\n",
        "ax.set_xlabel('Distance'); ax.set_ylabel('Density')\n",
        "ax.set_title('LBPH Distance Distribution'); ax.legend(); ax.grid(True)\n",
        "plt.tight_layout()\n",
        "plt.savefig(LOGS_DIR / 'confidence_distribution_analysis.png', dpi=150)\n",
        "plt.show()\n"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": ["## Existing Visualizations\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "plots_dir = LOGS_DIR / 'plots'\n",
        "if plots_dir.exists():\n",
        "    for img in plots_dir.glob('*.png'): display(Image(filename=str(img), width=700))\n"
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "summary = {'model': 'LBPH', 'params': metadata['model_params'],\n",
        "           'num_classes': metadata['num_classes'], 'threshold': metadata['threshold'],\n",
        "           'test_accuracy': metadata['test_accuracy']*100, 'test_coverage': metadata['test_coverage']*100}\n",
        "with open(LOGS_DIR / 'comprehensive_summary.json', 'w') as f: json.dump(summary, f, indent=2)\n",
        "print(json.dumps(summary, indent=2))\n"
    ]}
]

facenet_nb = {"cells": facenet_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 4}
lbph_nb = {"cells": lbph_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 4}

with open(r'd:\HCMUTE_project\DIP\FaceRecognition\notebooks\analysis_facenet_comprehensive.ipynb', 'w', encoding='utf-8') as f:
    json.dump(facenet_nb, f, indent=1)
print('FaceNet notebook created!')

with open(r'd:\HCMUTE_project\DIP\FaceRecognition\notebooks\analysis_lbph_comprehensive.ipynb', 'w', encoding='utf-8') as f:
    json.dump(lbph_nb, f, indent=1)
print('LBPH notebook created!')
