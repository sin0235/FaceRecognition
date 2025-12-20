import json
import os

# Đọc file Python
py_file = 'd:/HCMUTE_project/DIP/FaceRecognition/notebooks/evaluate_lbph_kaggle_optimized.py'
with open(py_file, 'r', encoding='utf-8') as f:
    content = f.read()

cells = []
current_lines = []
cell_type = 'code'
in_markdown = False

for line in content.split('\n'):
    stripped = line.strip()
    
    # Cell markdown marker
    if stripped.startswith('# %% [markdown]'):
        # Save current cell
        if current_lines:
            if cell_type == 'code':
                cells.append({
                    'cell_type': 'code',
                    'execution_count': None,
                    'metadata': {},
                    'outputs': [],
                    'source': [l + '\n' for l in current_lines[:-1]] + [current_lines[-1]] if current_lines else []
                })
            else:
                cells.append({
                    'cell_type': 'markdown',
                    'metadata': {},
                    'source': [l + '\n' for l in current_lines[:-1]] + [current_lines[-1]] if current_lines else []
                })
        current_lines = []
        cell_type = 'markdown'
        continue
    
    # Code cell marker
    if stripped == '# %%':
        # Save current cell
        if current_lines:
            if cell_type == 'code':
                cells.append({
                    'cell_type': 'code',
                    'execution_count': None,
                    'metadata': {},
                    'outputs': [],
                    'source': [l + '\n' for l in current_lines[:-1]] + [current_lines[-1]] if current_lines else []
                })
            else:
                cells.append({
                    'cell_type': 'markdown',
                    'metadata': {},
                    'source': [l + '\n' for l in current_lines[:-1]] + [current_lines[-1]] if current_lines else []
                })
        current_lines = []
        cell_type = 'code'
        continue
    
    # Remove leading "# " for markdown
    if cell_type == 'markdown' and line.startswith('# '):
        current_lines.append(line[2:])
    else:
        current_lines.append(line)

# Save last cell
if current_lines:
    if cell_type == 'code':
        cells.append({
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [l + '\n' for l in current_lines[:-1]] + [current_lines[-1]] if current_lines else []
        })
    else:
        cells.append({
            'cell_type': 'markdown',
            'metadata': {},
            'source': [l + '\n' for l in current_lines[:-1]] + [current_lines[-1]] if current_lines else []
        })

# Filter empty cells
cells = [c for c in cells if c['source'] and not all(s.strip() == '' for s in c['source'])]

notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.10.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

out_file = 'd:/HCMUTE_project/DIP/FaceRecognition/notebooks/evaluate_lbph_kaggle_optimized.ipynb'
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f'Created: {out_file}')
print(f'Total cells: {len(cells)}')
