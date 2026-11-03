import os
import json,sys
ROOT = os.environ.get("LOCUS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def norm(s,ci): s=" ".join(s.split()); return s.lower() if ci else s
for ds in sys.argv[1:]:
    man={r["id"]:r for r in map(json.loads,open(f"{ROOT}/data/{ds}/manifest.jsonl"))}
    for e in ["tess","trocr"]:
        try: H=[json.loads(l) for l in open(f"{ROOT}/data/{ds}/hyp_{e}.jsonl")]
        except FileNotFoundError: continue
        for ci in [False,True]:
            err={s:[0,0] for s in ["train","test"]}
            for h in H:
                s=man[h["id"]]["split"]; err[s][1]+=1; err[s][0]+= norm(h["hyp"],ci)!=norm(man[h["id"]]["gt"],ci)
            print(ds,e,"ci" if ci else "cs",{s:f"{a/b:.3f} of {b}" for s,(a,b) in err.items()})
    ex=[(man[h["id"]]["gt"],h["hyp"]) for h in H[:8]]; print(ex)
