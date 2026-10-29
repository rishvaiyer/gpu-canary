#!/usr/bin/env python3
"""
run_ref.py  --  Record one hardware/software stack's answers to the probe bank.

For every probe we compute the greedy next token and a few numeric fingerprints:
  - argmax_id        : which token wins (the 1-bit border measurement)
  - flip             : did argmax differ from the bank's reference argmax?
  - top2_gap         : current top-2 logit gap (fp precision leaves a trace here)
  - argmax_logit     : raw logit of the winner
  - logit_checksum   : sum of the full logit vector, rounded (coarse stack hash)

The resulting signature vector (one argmax id per probe) is what the detector
compares against later scans of the same stack. Different GPU/dtype/runtime stacks produce
measurably different signatures on the tight-gap probes.

Usage:
    python run_ref.py --bank ../probes/bank.json --label "m1-cpu-fp32" \
                      --dtype float32 --device cpu --out ../refs/sig_m1_cpu_fp32.json
"""
import argparse, json, os, platform, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=os.path.join(os.path.dirname(__file__), "..", "probes", "bank.json"))
    ap.add_argument("--label", required=True, help="human name for this stack")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.bank) as f:
        bank = json.load(f)
    model_name = bank["model"]
    dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dt).to(args.device)
    model.eval()

    sig, telemetry, flips = [], [], 0
    t0 = time.time()
    for pr in bank["probes"]:
        ids = tok(pr["prompt"], return_tensors="pt").to(args.device)
        with torch.no_grad():
            logits = model(**ids).logits[0, -1].float().cpu()
        top = torch.topk(logits, 2)
        # Decode rule must match the serving path (HF generate: argmax on
        # float32-cast logits). topk and argmax break exact ties differently, and
        # in bf16 many border probes ARE exact ties.
        amax = int(torch.argmax(logits))
        flip = amax != pr["argmax_id"]
        flips += int(flip)
        sig.append(amax)
        telemetry.append({
            "prompt": pr["prompt"],
            "argmax_id": amax,
            "argmax_tok": tok.decode(amax),
            "flip": flip,
            "top2_gap": round((top.values[0] - top.values[1]).item(), 6),
            "argmax_logit": round(top.values[0].item(), 6),
            "logit_checksum": round(logits.double().sum().item(), 3),
        })
    dt_s = time.time() - t0

    out = {
        "label": args.label,
        "model": model_name,
        "dtype": args.dtype,
        "device": args.device,
        "host": platform.node(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "n_probes": len(sig),
        "flips_vs_bank": flips,
        "seconds": round(dt_s, 3),
        "tok_per_s": round(len(sig) / dt_s, 2),
        "signature": sig,
        "telemetry": telemetry,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{args.label}] {len(sig)} probes, {flips} flips vs bank, "
          f"{out['tok_per_s']} probe/s on {out['host']}/{out['machine']} -> {args.out}")


if __name__ == "__main__":
    main()
