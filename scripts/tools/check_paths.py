import json, re, sys
LIT = re.compile(r"""(?<![\w])(f?)(["'])([^"'\n]*(?:\.csv|\.json|\.txt|\.dat|\.mtx|\.fsgd|\.mgh|\.mgz|\.png|\.gif|\.stats|/)[^"'\n]*)\2""")
for nb in sys.argv[1:]:
    print("=====", nb.split('/')[-1])
    for i, c in enumerate(json.load(open(nb))["cells"]):
        if c["cell_type"] != "code": continue
        t = "".join(c["source"])
        if t.lstrip().startswith("%%") or i == 0: continue
        for ln in t.splitlines():
            if ln.lstrip().startswith("#"): continue
            for m in LIT.finditer(ln):
                pre = ln[:m.start()]
                s = m.group(3)
                if re.search(r"(inp|out)\(\s*$", pre) or "{DATA}" in s or re.search(r"(DATA\)?\s*\+|DATA\s*/)\s*$", pre): continue
                if s.strip() in ("/", " / ") or "<" in s or "{male" in s or "timedelta" in s: continue
                print(f"  [{i}] {ln.strip()[:150]}")
