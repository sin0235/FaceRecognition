import json
import sys

def analyze_notebook(notebook_path):
    """Phân tích notebook và tìm các function calls cần update."""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    issues = []
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
            
        source = ''.join(cell['source'])
        
        # Tìm các function calls cũ
        if 'train_classifier' in source:
            issues.append({
                'cell': i + 1,
                'type': 'function_renamed',
                'old': 'train_classifier',
                'new': 'train_lbph_model',
                'snippet': source[:200]
            })
        
        # Kiểm tra unpacking của evaluate_lbph
        if 'evaluate_lbph' in source:
            # Old signature: acc, reject_rate, confs, used, coverage
            # New signature: accuracy, coverage, used, confidences
            if 'reject' in source or ', _, ' in source:
                issues.append({
                    'cell': i + 1,
                    'type': 'return_values_changed',
                    'function': 'evaluate_lbph',
                    'snippet': source[:200]
                })
        
        # Kiểm tra find_best_threshold
        if 'find_best_threshold' in source:
            issues.append({
                'cell': i + 1,
                'type': 'function_renamed',
                'old': 'find_best_threshold',
                'new': 'find_optimal_threshold',
                'snippet': source[:200]
            })
    
    return issues

if __name__ == '__main__':
    notebooks = [
        'notebooks/train_lbph.ipynb',
        'notebooks/evaluate_lbph.ipynb',
        'notebooks/prepare_lbph_dataset.ipynb'
    ]
    
    for nb_path in notebooks:
        print(f"\n{'='*70}")
        print(f"Analyzing: {nb_path}")
        print('='*70)
        
        try:
            issues = analyze_notebook(nb_path)
            
            if not issues:
                print("✓ No issues found - notebook is compatible")
            else:
                print(f"⚠ Found {len(issues)} issues:\n")
                for issue in issues:
                    print(f"Cell {issue['cell']}:")
                    print(f"  Type: {issue['type']}")
                    if 'old' in issue:
                        print(f"  Change: {issue['old']} → {issue['new']}")
                    print(f"  Snippet: {issue['snippet'][:100]}...")
                    print()
                    
        except FileNotFoundError:
            print(f"✗ File not found")
        except Exception as e:
            print(f"✗ Error: {e}")
