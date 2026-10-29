#!/usr/bin/env python3
"""
faultsim.py -- Emulate PUBLISHED GPU silent-data-corruption patterns and measure
whether border probes catch them vs ordinary prompts.

Fault patterns (from SDCHunter OSDI'26, AEGIS OSDI'26, Tung et al. 2026):
  zero     one output element of a Linear layer is nullified (50.7% of corruptions)
  sign     one element gets a flipped sign (seen in AEGIS incident)
  bitflip  one element gets a random mantissa/exponent bit flipped
  bias     one element gets a relative bias of 1e-3 (SDCHunter "GPU 2", subtle)
Each Linear forward is corrupted with probability RATE (intermittent faults).
Position is random per forward. We repeat the whole 64-probe scan REPS times and
report how often a scan flags at least one mismatch (scan-level recall), for
border probes and for ordinary prompts.
"""
import argparse, json, os, random, sys, time, warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT,"probes"))
from make_probes import SEED_PROMPTS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bank", default=os.path.join(ROOT,"probes","bank.json"))
    ap.add_argument("--dtype", default="float32", choices=["float32","float16","bfloat16"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fault", required=True, choices=["zero","sign","bitflip","bias"])
    ap.add_argument("--rate", type=float, required=True, help="P(corrupt) per Linear forward")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None)
    a=ap.parse_args()
    rng=random.Random(a.seed)
    bank=json.load(open(a.bank))
    dt={"float32":torch.float32,"float16":torch.float16,"bfloat16":torch.bfloat16}[a.dtype]
    tok=AutoTokenizer.from_pretrained(bank["model"])
    model=AutoModelForCausalLM.from_pretrained(bank["model"], dtype=dt).to(a.device).eval()
    border=[p["prompt"] for p in bank["probes"]]; ordinary=list(SEED_PROMPTS)

    state={"on":False,"hits":0}
    def corrupt(mod, inp, out):
        if not state["on"] or rng.random()>=a.rate: return out
        out=out.clone(); flat=out.view(-1); i=rng.randrange(flat.numel()); v=flat[i]
        if a.fault=="zero": flat[i]=0
        elif a.fault=="sign": flat[i]=-v
        elif a.fault=="bias": flat[i]=v*(1+1e-3)
        elif a.fault=="bitflip":
            f=v.detach().float().view(torch.int32).item()
            bit=rng.randrange(0,31)          # any mantissa/exponent bit, never sign
            f^=(1<<bit)
            flat[i]=torch.tensor(f,dtype=torch.int32).view(torch.float32).to(out.dtype)
        state["hits"]+=1
        return out
    hooks=[m.register_forward_hook(corrupt) for m in model.modules() if isinstance(m, torch.nn.Linear)]

    def run(prompts):
        out=[]
        for p in prompts:
            ids=tok(p,return_tensors="pt").to(a.device)
            with torch.no_grad(): lg=model(**ids).logits[0,-1].float().cpu()
            out.append(int(torch.argmax(lg)))
        return out
    state["on"]=False
    ref_b=run(border); ref_o=run(ordinary)        # clean reference, same process
    res={"fault":a.fault,"rate":a.rate,"dtype":a.dtype,"device":a.device,"reps":a.reps,
         "border_scans_flagged":0,"ordinary_scans_flagged":0,"border_probe_flips":0,"ordinary_prompt_flips":0,"corruptions":0}
    t0=time.time()
    for r in range(a.reps):
        state["on"]=True; state["hits"]=0
        b=run(border); fb=sum(x!=y for x,y in zip(b,ref_b))
        o=run(ordinary); fo=sum(x!=y for x,y in zip(o,ref_o))
        res["corruptions"]+=state["hits"]
        res["border_probe_flips"]+=fb; res["ordinary_prompt_flips"]+=fo
        res["border_scans_flagged"]+=int(fb>0); res["ordinary_scans_flagged"]+=int(fo>0)
    res["seconds"]=round(time.time()-t0,1)
    res["border_recall"]=round(res["border_scans_flagged"]/a.reps,2)
    res["ordinary_recall"]=round(res["ordinary_scans_flagged"]/a.reps,2)
    line=json.dumps(res); print(line)
    if a.log: open(a.log,"a").write(line+"\n")
if __name__=="__main__": main()
