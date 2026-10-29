#!/usr/bin/env python3
"""
persist.py -- Persistent (deterministic) hardware fault vs two detection styles.

A degraded unit that always computes the same wrong value (SDCHunter: replay
occurrence rate 100% on several GPUs; errors escalate over time) is invisible
to self-consistency checks (run twice, compare bitwise: AEGIS-style redundant
execution), because both runs are wrong identically. A reference-based border
probe check sees it, because the reference was recorded when the unit was healthy.

Fault: a FIXED element (layer L, output index i) of one Linear is corrupted the
same way on every forward. Patterns: zero, sign, bias1e-3, bias1e-2.
Reports: border mismatches vs saved healthy reference; self-check differences.
"""
import argparse, json, os, random, sys, warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT,"probes"))
from make_probes import SEED_PROMPTS

ap=argparse.ArgumentParser()
ap.add_argument("--bank", default=os.path.join(ROOT,"probes","bank.json"))
ap.add_argument("--dtype", default="float32"); ap.add_argument("--device", default="cpu")
ap.add_argument("--trials", type=int, default=8, help="random fixed fault sites per pattern")
ap.add_argument("--log", default=None)
a=ap.parse_args()
dt={"float32":torch.float32,"float16":torch.float16,"bfloat16":torch.bfloat16}[a.dtype]
bank=json.load(open(a.bank)); tok=AutoTokenizer.from_pretrained(bank["model"])
model=AutoModelForCausalLM.from_pretrained(bank["model"], dtype=dt).to(a.device).eval()
border=[p["prompt"] for p in bank["probes"]]; ordinary=list(SEED_PROMPTS)
linears=[m for m in model.model.layers for m in m.modules() if isinstance(m, torch.nn.Linear)]

def run(prompts):
    out=[]
    for p in prompts:
        ids=tok(p,return_tensors="pt").to(a.device)
        with torch.no_grad(): out.append(int(torch.argmax(model(**ids).logits[0,-1].float().cpu())))
    return out
ref_b=run(border); ref_o=run(ordinary)
rng=random.Random(1)
for pat in ["zero","sign","bias1e-3","bias1e-2"]:
    res={"pattern":pat,"trials":a.trials,"border_flagged":0,"ordinary_flagged":0,"selfcheck_flagged":0,"border_flips":0}
    for t in range(a.trials):
        lin=rng.choice(linears); idx=rng.randrange(lin.out_features)
        def hook(m,i,o,idx=idx):
            o=o.clone(); v=o[...,idx]
            if pat=="zero": o[...,idx]=0
            elif pat=="sign": o[...,idx]=-v
            elif pat=="bias1e-3": o[...,idx]=v*1.001
            elif pat=="bias1e-2": o[...,idx]=v*1.01
            return o
        h=lin.register_forward_hook(hook)
        b1=run(border); b2=run(border); o1=run(ordinary)
        h.remove()
        fb=sum(x!=y for x,y in zip(b1,ref_b)); fo=sum(x!=y for x,y in zip(o1,ref_o))
        sc=sum(x!=y for x,y in zip(b1,b2))           # self-consistency on the same probes
        res["border_flips"]+=fb; res["border_flagged"]+=int(fb>0); res["ordinary_flagged"]+=int(fo>0); res["selfcheck_flagged"]+=int(sc>0)
    line=json.dumps(res); print(line, flush=True)
    if a.log: open(a.log,"a").write(line+"\n")
