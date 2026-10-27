"""Local rating page for the QC snapshots: outputs/qc/index.html (open it next to outputs/qc/png/).

Each image gets pass / fail buttons, the failing version(s) and a comment; ratings are kept in the browser
(localStorage) and exported with the "Download ratings.csv" button (same columns as outputs/qc/ratings.csv).
Participant-level images: keep the page local, never publish it.
Usage: python scripts/qc/make_qc_page.py
"""
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
import fsvv  # noqa: E402

QC = fsvv.OUT / "qc"
images = sorted(p.stem for p in (QC / "png").glob("*.png"))
rows = "\n".join(
    f'<section id="{html.escape(i)}"><h3>{html.escape(i)}</h3><img src="png/{html.escape(i)}.png" loading="lazy">'
    f'<div><label><input type="radio" name="{i}" value="pass"> pass</label> '
    f'<label><input type="radio" name="{i}" value="fail"> fail</label> '
    f'version(s) <input class="v" size="12"> comment <input class="c" size="50"></div></section>'
    for i in images)
page = f"""<!doctype html><html><head><meta charset="utf-8"><title>FreeSurfer QC</title><style>
body{{font-family:sans-serif;margin:16px;background:#fff;color:#111}} img{{max-width:100%}}
section{{border-bottom:1px solid #ccc;padding:8px 0}} #bar{{position:sticky;top:0;background:#eee;padding:6px}}
</style></head><body><div id="bar"><b>{len(images)} images</b> — rated: <span id="n">0</span>
<button onclick="dl()">Download ratings.csv</button></div>
<p>Rate each image (rows: FS5–FS8). Fail = segmentation or surface error that would change the regional measures.</p>
{rows}
<script>
const K='fsvv-qc';let R={{}};try{{R=JSON.parse(localStorage.getItem(K)||'{{}}')}}catch(e){{}}
function save(){{try{{localStorage.setItem(K,JSON.stringify(R))}}catch(e){{}};document.getElementById('n').textContent=Object.values(R).filter(r=>r.rating).length}}
document.querySelectorAll('section').forEach(s=>{{const id=s.id,r=R[id]||{{}};
 s.querySelectorAll('input[type=radio]').forEach(b=>{{b.checked=b.value===r.rating;b.onchange=()=>{{(R[id]=R[id]||{{}}).rating=b.value;save()}}}});
 const v=s.querySelector('.v'),c=s.querySelector('.c');v.value=r.version||'';c.value=r.comment||'';
 v.oninput=()=>{{(R[id]=R[id]||{{}}).version=v.value;save()}};c.oninput=()=>{{(R[id]=R[id]||{{}}).comment=c.value;save()}}}});
save();
function q(x){{return '"'+String(x||'').replace(/"/g,'""')+'"'}}
function dl(){{const lines=['image,rating,version,comment'];document.querySelectorAll('section').forEach(s=>{{const r=R[s.id]||{{}};lines.push([s.id,r.rating,r.version,r.comment].map(q).join(','))}});
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\n')],{{type:'text/csv'}}));a.download='ratings.csv';a.click()}}
</script></body></html>"""
(QC / "index.html").write_text(page)
print("wrote", QC / "index.html", len(images), "images")
