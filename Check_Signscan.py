#!/usr/bin/env python3
"""
Audita significance_scan en cada notebook.
Encuentra TODAS las definiciones (no solo la primera) y compara la LOGICA via AST
normalizado, que ignora comentarios, docstrings y formato.

Uso:  python3 check_sigscan.py Final_Notebooks/*.ipynb
"""
import ast, json, sys, hashlib, textwrap
from pathlib import Path
from collections import defaultdict

FN = "significance_scan"

def find_all(nb_path):
    """Devuelve [(cell_idx, src)] para CADA def significance_scan del notebook."""
    nb = json.load(open(nb_path))
    hits = []
    for ci, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == FN:
                lines = src.splitlines()
                seg = textwrap.dedent("\n".join(lines[node.lineno-1:node.end_lineno]))
                hits.append((ci, seg))
    return hits

def logic_fp(src):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.Module, ast.ClassDef)):
            b = getattr(node, "body", None)
            if (b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant)
                    and isinstance(b[0].value.value, str)):
                node.body = b[1:]
    return hashlib.md5(ast.dump(tree).encode()).hexdigest()[:8]

rows, logics, dupes = [], defaultdict(list), []
for p in sys.argv[1:]:
    name = Path(p).stem
    hits = find_all(p)
    if not hits:
        rows.append((name, "-", "-", "-", "NO ENCONTRADA"))
        continue
    if len(hits) > 1:
        dupes.append((name, [c for c, _ in hits]))
    for ci, src in hits:
        txt = hashlib.md5(src.encode()).hexdigest()[:8]
        try:
            lg = logic_fp(src)
        except SyntaxError:
            lg = "PARSE-ERR"
        style = "vectorizada" if "cumsum" in src else "loop naive"
        nan_ok = "isfinite" in src or "isnan" in src
        tag = f"{style}{'' if nan_ok else '  [SIN guarda NaN]'}"
        rows.append((f"{name}", f"celda {ci}", txt, lg, tag))
        logics[lg].append(f"{name}#c{ci}")

w0 = max(len(r[0]) for r in rows) + 2
w1 = max(len(r[1]) for r in rows) + 2
print(f"{'notebook':<{w0}}{'donde':<{w1}}{'texto':<10}{'logica':<10}estilo")
print("-" * (w0 + w1 + 40))
for r in rows:
    print(f"{r[0]:<{w0}}{r[1]:<{w1}}{r[2]:<10}{r[3]:<10}{r[4]}")

print()
if dupes:
    print("!! DEFINICIONES DUPLICADAS (la ultima ejecutada gana - shadowing):")
    for n, cells in dupes:
        print(f"   {n}: celdas {cells}")
    print()
print(f"Grupos de logica distinta: {len(logics)}")
for i, (lg, names) in enumerate(logics.items(), 1):
    print(f"  [{i}] {lg}: {', '.join(names)}")