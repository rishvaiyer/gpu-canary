#!/usr/bin/env python3
"""
health/check.py -- Silent-corruption alarm using border probes.

Runs the probe bank on this machine and compares the argmax tokens to the saved
reference signature for the same stack. Healthy hardware reproduces the reference
exactly. Any mismatch means the numerics drifted: a failing card, a changed
library, or a changed kernel path.

--inject LAYER:LANE:EPS simulates a silent fault: one output lane of one Linear
layer is multiplied by (1+EPS) on every forward. Used to measure how small a
fault the probes can see, versus ordinary prompts (--ordinary).
"""
import argparse, json, os, platform, sys, time, warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "probes"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=os.path.join(ROOT, "probes", "bank.json"))
    ap.add_argument("--ref", required=True, help="refs/sig_*.json for this stack")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--inject", default=None, help="LAYER_IDX:LANE:EPS  e.g. 20:7:1e-4")
    ap.add_argument("--ordinary", action="store_true",
                    help="use the ordinary seed prompts instead of border probes (control)")
    ap.add_argument("--log", default=None, help="append one JSON line here")
    args = ap.parse_args()

    bank = json.load(open(args.bank))
    ref = json.load(open(args.ref))
    dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    tok = AutoTokenizer.from_pretrained(bank["model"])
    model = AutoModelForCausalLM.from_pretrained(bank["model"], dtype=dt).to(args.device).eval()

    if args.ordinary:
        from make_probes import SEED_PROMPTS as SEEDS
        prompts = list(SEEDS)
    else:
        prompts = [p["prompt"] for p in bank["probes"]]

    hook = None
    if args.inject:
        li, lane, eps = args.inject.split(":"); li, lane, eps = int(li), int(lane), float(eps)
        layer = model.model.layers[li].mlp.down_proj
        def corrupt(mod, inp, out):
            out = out.clone(); out[..., lane] = out[..., lane] * (1.0 + eps); return out
        hook = layer.register_forward_hook(corrupt)

    def argmax_tok(prompt):
        ids = tok(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            logits = model(**ids).logits[0, -1].float().cpu()
        return tok.decode(int(torch.argmax(logits)))

    t0 = time.time()
    got = [argmax_tok(p) for p in prompts]
    if hook: hook.remove()

    if args.ordinary:
        # reference for ordinary prompts = this same model, clean, computed now
        clean = got if not args.inject else None
        if clean is None:
            expect = [argmax_tok(p) for p in prompts]
        else:
            expect = clean
    else:
        expect = [t["argmax_tok"] for t in ref["telemetry"]]

    bad = [i for i, (a, b) in enumerate(zip(got, expect)) if a != b]
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(), "dtype": args.dtype, "device": args.device,
        "set": "ordinary" if args.ordinary else "border",
        "inject": args.inject, "n": len(prompts), "mismatches": len(bad), "bad_idx": bad,
        "seconds": round(time.time() - t0, 1),
        "status": "OK" if not bad else "DRIFT",
    }
    line = json.dumps(rec)
    print(line)
    if args.log:
        with open(args.log, "a") as f: f.write(line + "\n")
    sys.exit(0 if not bad else 2)

if __name__ == "__main__":
    main()
