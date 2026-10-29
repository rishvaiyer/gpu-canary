#!/usr/bin/env python3
"""
prism_fault.py -- Permanent GPU faults modelled after LLM-PRISM (Tyagi et al., arXiv:2604.10390,
NVIDIA / U. Rochester, 2026), the closest published method for *permanent* faults in LLM
computation, whose error signatures come from RTL-level fault simulation of an H100.

Their model, reproduced here as faithfully as the paper allows:
  - A defect lives on one physical device at one fixed location. We pin it to one Linear
    layer and one fixed output tile, the same on every forward.
  - Unit types, with RTL error signatures (quoted from the paper):
      tensorcore : "localized 3x3 patches in the matrix where all corruptions share the
                    same flipped bit position"
      fma        : "a small number of operations exhibit sporadic, high-magnitude numeric
                    outliers"
      l1cache    : "an entire matrix row's stored values are corrupted when accessed",
                    bit distribution "centered on high-order bits"
  - Fault density: the proportion of corrupted elements inside the target tile.
  - Error rate r in (0,1]: per-forward Bernoulli probability that the defect produces a
    visible corruption; r = 1.0 models a hard stuck-at defect.
  - Single-bit flips are weighted more heavily than multi-bit patterns.
  - Units sampled by silicon area. The paper does not give the areas; we use
    tensorcore 0.5, fma 0.3, l1cache 0.2 and state this as an assumption.

Everything else (models, probes, scans, controls, redundant execution) is as in persist.py.
Output: one JSON line per (unit, rate): sites flagged by border probes, by ordinary prompts,
and by run-twice self-check, over N random sites.
"""
import argparse, json, os, random, sys, warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "probes"))
from make_probes import SEED_PROMPTS

UNIT_AREA = {"tensorcore": 0.5, "fma": 0.3, "l1cache": 0.2}

def flip_bits(x, bits):
    """Flip the given fp16/bf16/fp32 bit positions of tensor x (in place semantics via return)."""
    if x.dtype == torch.float16:
        i = x.view(torch.int16); m = 0
        for b in bits: m ^= (1 << b)
        return (i ^ torch.tensor(m, dtype=torch.int16, device=x.device)).view(torch.float16)
    if x.dtype == torch.bfloat16:
        i = x.view(torch.int16); m = 0
        for b in bits: m ^= (1 << b)
        return (i ^ torch.tensor(m, dtype=torch.int16, device=x.device)).view(torch.bfloat16)
    i = x.view(torch.int32); m = 0
    for b in bits: m ^= (1 << b)
    return (i ^ torch.tensor(m, dtype=torch.int32, device=x.device)).view(torch.float32)

def nbits(dtype):  # non-sign bits
    return 15 if dtype in (torch.float16, torch.bfloat16) else 31

class PrismDefect:
    """One permanent defect: unit type, fixed layer, fixed tile, fixed bit pattern."""
    def __init__(self, unit, layer, out_features, rng, dtype, rate, density, tile=16):
        self.unit, self.layer, self.rate, self.density, self.tile = unit, layer, rate, density, tile
        self.rng = rng
        self.c0 = rng.randrange(0, max(1, out_features - tile))      # tile column start
        self.r0 = rng.randrange(0, 8)                                  # tile row start (token index)
        nb = nbits(dtype)
        # single-bit weighted 3:1 over double-bit
        k = 1 if rng.random() < 0.75 else 2
        if unit == "l1cache":   # centred on high-order bits: exponent region
            hi = range(nb - 5, nb)
            self.bits = rng.sample(list(hi), k)
        else:
            self.bits = rng.sample(range(0, nb), k)
        self.hits = 0
        # Fixed corruption pattern per defect: a stuck unit corrupts the same elements every time.
        g = torch.Generator().manual_seed(rng.randrange(1 << 30))
        self.patch_mask = torch.rand((3, 3), generator=g) < max(density, 1e-9)
        if not self.patch_mask.any(): self.patch_mask[1, 1] = True
        self.row_mask = torch.rand((out_features,), generator=g) < density
        self.fma_thresh = int(density * 0.05 * 1024)   # sporadic outliers, decided by the value bits
    def hook(self, mod, inp, out):
        if self.rng.random() >= self.rate: return out
        out = out.clone(); L, N = out.shape[-2], out.shape[-1]
        c0 = min(self.c0, N - self.tile); r0 = min(self.r0, max(0, L - 1))
        self.hits += 1
        if self.unit == "tensorcore":
            rows = slice(r0, min(L, r0 + 3)); cols = slice(c0, c0 + 3)
            patch = out[..., rows, cols]
            mask = self.patch_mask[: patch.shape[-2], : patch.shape[-1]].to(out.device)
            out[..., rows, cols] = torch.where(mask, flip_bits(patch, self.bits), patch)
        elif self.unit == "fma":
            rows = slice(r0, min(L, r0 + self.tile)); cols = slice(c0, c0 + self.tile)
            patch = out[..., rows, cols]
            # sporadic, input-dependent: the low mantissa bits of the operand decide whether this
            # operation trips the defect and how large the outlier is (deterministic for a given input)
            bits = patch.float().view(torch.int32) & 0x3FF
            mask = bits < self.fma_thresh
            scale = torch.pow(2.0, 8 + (bits % 8).float()).to(patch.dtype)
            sign = torch.where((bits & 1) == 1, -1.0, 1.0).to(patch.dtype)
            out[..., rows, cols] = torch.where(mask, patch * scale * sign, patch)
        elif self.unit == "l1cache":
            row = out[..., r0, :]
            mask = self.row_mask[: row.shape[-1]].to(out.device)
            out[..., r0, :] = torch.where(mask, flip_bits(row, self.bits), row)
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=os.path.join(ROOT, "probes", "bank.json"))
    ap.add_argument("--dtype", default="float16"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--sites", type=int, default=8); ap.add_argument("--density", type=float, default=0.5)
    ap.add_argument("--rates", default="1.0,0.1"); ap.add_argument("--units", default="tensorcore,fma,l1cache")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log", default=None)
    a = ap.parse_args()
    dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[a.dtype]
    bank = json.load(open(a.bank)); tok = AutoTokenizer.from_pretrained(bank["model"])
    model = AutoModelForCausalLM.from_pretrained(bank["model"], dtype=dt).to(a.device).eval()
    border = [p["prompt"] for p in bank["probes"]]; ordinary = list(SEED_PROMPTS)
    layers = list(model.model.layers)
    rng = random.Random(a.seed)

    def run(prompts):
        out = []
        for p in prompts:
            ids = tok(p, return_tensors="pt").to(a.device)
            with torch.no_grad(): out.append(int(torch.argmax(model(**ids).logits[0, -1].float().cpu())))
        return out
    ref_b = run(border); ref_o = run(ordinary)

    for unit in a.units.split(","):
        for rate in [float(x) for x in a.rates.split(",")]:
            res = {"model": bank["model"], "unit": unit, "rate": rate, "density": a.density, "sites": a.sites,
                   "dtype": a.dtype, "border_flagged": 0, "ordinary_flagged": 0, "selfcheck_flagged": 0,
                   "border_flips": 0, "ordinary_flips": 0, "corruptions": 0, "area_weight": UNIT_AREA[unit]}
            for s in range(a.sites):
                li = rng.randrange(len(layers))
                lins = [m for m in layers[li].modules() if isinstance(m, torch.nn.Linear)]
                lin = rng.choice(lins)
                d = PrismDefect(unit, li, lin.out_features, rng, dt, rate, a.density)
                h = lin.register_forward_hook(d.hook)
                b1 = run(border); b2 = run(border); o1 = run(ordinary)
                h.remove()
                fb = sum(x != y for x, y in zip(b1, ref_b)); fo = sum(x != y for x, y in zip(o1, ref_o))
                sc = sum(x != y for x, y in zip(b1, b2))
                res["border_flips"] += fb; res["ordinary_flips"] += fo; res["corruptions"] += d.hits
                res["border_flagged"] += int(fb > 0); res["ordinary_flagged"] += int(fo > 0); res["selfcheck_flagged"] += int(sc > 0)
            line = json.dumps(res); print(line, flush=True)
            if a.log: open(a.log, "a").write(line + "\n")

if __name__ == "__main__": main()
