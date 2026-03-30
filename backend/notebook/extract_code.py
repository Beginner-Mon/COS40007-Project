import json
import os

files = [
    'boniong_vs_slicing.ipynb',
    'knife_sharpness_3class.ipynb',
    'activity_recognition.ipynb'
]

for file in files:
    try:
        with open(file, 'r', encoding='utf-8-sig') as f:
            nb = json.load(f)
            code_cells = [
                ''.join(cell.get('source', [])) 
                for cell in nb.get('cells', []) 
                if cell.get('cell_type') == 'code'
            ]
            code = '\n\n# --- CELL ---\n\n'.join(code_cells)
            
            with open(file.replace('.ipynb', '.py'), 'w', encoding='utf-8') as out:
                out.write(code)
            print(f"Successfully extracted {file}")
    except Exception as e:
        print(f"Failed to process {file}: {e}")
