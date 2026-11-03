#!/bin/bash
LOCUS_ROOT="${LOCUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Final build: dedupe result files -> regenerate every table/figure -> compile the paper.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ml_env
cd ${LOCUS_ROOT}
python3 - <<'EOF'
# Duplicate rows can arise when two lanes ran the same deterministic step; keep the last copy.
import json, glob
keys = {"locus_main": ("ds", "method", "k", "seed"), "external": ("ds", "method", "k", "seed"),
        "locus_ablate": ("ds", "variant", "k", "seed"), "locus_xengine": ("ds", "train", "test", "k", "seed", "setting"),
        "locus_xdataset": ("train", "test", "seed"), "locus_ablate_single": ("ds", "variant", "k", "seed"),
        "locus_sensitivity": ("ds", "factor", "value")}
for name, key in keys.items():
    p = f"results/{name}.jsonl"
    try:
        rows = [json.loads(l) for l in open(p)]
    except FileNotFoundError:
        continue
    d = {}
    for r in rows:
        d[tuple(str(r.get(k)) for k in key)] = r
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in d.values()))
    print(f"{name}: {len(rows)} -> {len(d)} rows")
EOF
python code/make_assets.py 2>&1 | grep -E "^ok|FAILED|incomplete"
python code/make_analysis.py 2>&1 | grep -E "^ok|FAILED"
python code/make_datafig.py 2>&1 | grep -i "Traceback"; python code/make_qual2.py 2>&1 | grep -i "Traceback" ; python -c "import sys; sys.path.insert(0,'code'); import make_qual2 as Q; Q.audit_table()"
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > ../logs/latex_final.log 2>&1; echo "latex exit $?"
grep -E "^!" -A3 main.log | head; grep -E "undefined" main.log | sort -u | head; pdfinfo main.pdf | grep Pages
