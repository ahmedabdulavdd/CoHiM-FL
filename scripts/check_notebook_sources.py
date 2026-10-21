#!/usr/bin/env python3
"""Static syntax/metadata checks for Jupyter sources; does not execute any code."""
from pathlib import Path
import json,ast
root=Path(__file__).resolve().parents[1]/'notebooks'
expected=[7,21,42,84,126]
found=[]
for path in sorted(root.glob('*.ipynb')):
    n=json.loads(path.read_text())
    assert n['nbformat']==4
    assert n.get('cells')
    for i,c in enumerate(n['cells']):
        if c['cell_type']!='code':continue
        src=''.join(c.get('source',[]))
        # IPython magic and shell commands are valid in Colab, not Python AST.
        if src.strip() and not any(line.lstrip().startswith(('!','%')) for line in src.splitlines()):
            try:ast.parse(src)
            except SyntaxError as e: raise SyntaxError(f'{path.name}, cell {i}: {e}')
    for seed in expected:
        if path.name.startswith(f'01_seed_{seed}_'):found.append(seed)
    print('PASS notebook JSON',path.name)
assert sorted(found)==expected
print('PASS all five independent-seed notebooks included')
