import ast, json, sys
for nb in sys.argv[1:]:
    bad = 0
    for i, c in enumerate(json.load(open(nb))["cells"]):
        t = "".join(c["source"])
        if c["cell_type"] != "code" or t.lstrip().startswith("%%"): continue
        code = "\n".join("" if l.lstrip().startswith(("%", "!")) else l for l in t.splitlines())
        try: ast.parse(code)
        except SyntaxError as e: bad += 1; print(f"  {nb.split('/')[-1]} cell {i}: {e.msg} line {e.lineno}: {code.splitlines()[e.lineno-1][:120] if e.lineno else ''}")
    print(nb.split('/')[-1], "syntax errors:", bad)
