#!/usr/bin/env python3
"""
Busca funciones definidas MAS DE UNA VEZ dentro de un mismo notebook (shadowing:
gana la celda ejecutada al final) y, opcionalmente, compara la logica de una
funcion entre notebooks via AST normalizado (ignora comentarios/docstrings/formato).

Uso:
  python3 check_dupes.py Final_Notebooks/*.ipynb
  python3 check_dupes.py --fn run_optuna_search Final_Notebooks/*.ipynb
  python3 check_dupes.py --fn significance_scan Final_Notebooks/*.ipynb
"""
import ast, json, sys, hashlib, textwrap, argparse
from pathlib import Path
from collections import defaultdict

def defs_in_notebook(p):
    """[(fname, cell_idx, src)] para cada def de nivel superior."""
    nb = json.load(open(p))
    out = []
    for ci, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        lines = src.splitlines()
        for node in tree.body:                      # solo nivel superior
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                seg = textwrap.dedent("\n".join(lines[node.lineno-1:node.end_lineno]))
                out.append((node.name, ci, seg))
    return out

def logic_fp(src):
    tree = ast.parse(src)
    for n in ast.walk(tree):
        b = getattr(n, "body", None)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) and b:
            if (isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant)
                    and isinstance(b[0].value.value, str)):
                n.body = b[1:]
    return hashlib.md5(ast.dump(tree).encode()).hexdigest()[:8]

ap = argparse.ArgumentParser()
ap.add_argument("--fn", default=None, help="comparar esta funcion entre notebooks")
ap.add_argument("files", nargs="+")
a = ap.parse_args()

print("=" * 70)
print("1) DUPLICADOS DENTRO DE UN MISMO NOTEBOOK (shadowing)")
print("=" * 70)
any_dupe = False
for p in a.files:
    seen = defaultdict(list)
    for name, ci, src in defs_in_notebook(p):
        seen[name].append((ci, src))
    dupes = {n: v for n, v in seen.items() if len(v) > 1}
    if dupes:
        any_dupe = True
        print(f"\n!! {Path(p).stem}")
        for n, occ in sorted(dupes.items()):
            fps = [logic_fp(s) for _, s in occ]
            same = "IDENTICAS" if len(set(fps)) == 1 else "LOGICA DISTINTA"
            cells = ", ".join(f"celda {c}" for c, _ in occ)
            print(f"   {n}: {len(occ)}x ({cells}) -> {same}  [{', '.join(fps)}]")
            print(f"      gana la celda {occ[-1][0]} si ejecutas de arriba a abajo")
if not any_dupe:
    print("\n(ninguno)")

if a.fn:
    print()
    print("=" * 70)
    print(f"2) LOGICA DE '{a.fn}' ENTRE NOTEBOOKS")
    print("=" * 70)
    groups = defaultdict(list)
    missing = []
    for p in a.files:
        hits = [(ci, s) for n, ci, s in defs_in_notebook(p) if n == a.fn]
        if not hits:
            missing.append(Path(p).stem); continue
        for ci, s in hits:
            groups[logic_fp(s)].append(f"{Path(p).stem}#c{ci}")
    for i, (fp, names) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1])), 1):
        print(f"  [{i}] {fp}  ({len(names)}): {', '.join(names)}")
    if missing:
        print(f"  no definida en: {', '.join(missing)}")
    print(f"\n  -> {len(groups)} implementacion(es) distinta(s); "
          f"{'CONSISTENTE' if len(groups) <= 1 else 'HAY DIVERGENCIA'}")