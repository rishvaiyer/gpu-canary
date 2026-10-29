#!/usr/bin/env python3
"""
make_probes.py  --  Build "border" probes for hardware/stack attribution.

Idea (from HSPI, arXiv:2411.05197): when the top-2 next-token logits of a model
are almost tied, floating-point rounding differences between hardware/software
stacks can flip which token wins under greedy decoding. A prompt whose next
token sits on such a knife-edge is a "border input". The argmax token at that
position is a 1-bit measurement of the serving stack.

This script scans a corpus of candidate prompts through ONE open-weights model
and keeps the prompts whose top-2 logit gap is smallest. Those are the probes we
later replay as a health check. We also record, per probe, the reference argmax
and the two candidate token ids, so the detector knows what a "flip" looks like.

Run the SAME script unchanged on each reference stack via run_ref.py; the probe
SET is chosen once (here) so every stack answers the identical questions.

Usage:
    python make_probes.py --model Qwen/Qwen2.5-0.5B --n 64 --out ../probes/bank.json
"""
import argparse, json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# A deterministic corpus of short continuation prompts. Variety matters more
# than meaning: we want many positions where two plausible tokens compete.
SEED_PROMPTS = [
    "The capital of the country north of France is",
    "2 + 2 =",
    "The opposite of hot is",
    "In 1492 Columbus sailed the ocean",
    "Water is made of hydrogen and",
    "The quick brown fox jumps over the lazy",
    "To be or not to",
    "The first president of the United States was George",
    "A group of lions is called a",
    "The chemical symbol for gold is",
    "The square root of 144 is",
    "Monday, Tuesday, Wednesday,",
    "Red, orange, yellow, green,",
    "The largest planet in the solar system is",
    "She opened the door and saw a",
    "The cat sat on the",
    "import numpy as",
    "def add(a, b): return a +",
    "The mitochondria is the powerhouse of the",
    "E equals m c",
    "Once upon a time there was a",
    "The author of Romeo and Juliet is William",
    "Paris is the capital of",
    "The speed of light is approximately 300,000 kilometers per",
    "Roses are red, violets are",
    "The three primary colors are red, blue, and",
    "An apple a day keeps the doctor",
    "The boiling point of water in Celsius is",
    "The sun rises in the",
    "A triangle has three",
    "The currency of Japan is the",
    "Neil Armstrong was the first man on the",
]


def candidate_prompts(tok, pool, seed=0):
    """Yield many candidate prompts: the hand seeds, plus random token windows.

    Natural sentences almost never land on a logit knife-edge (their top-2 gap is
    large), so harvesting them finds no usable border probes. HSPI crafts border
    inputs instead. We approximate that by sampling random short token sequences;
    a fraction of them leave the model genuinely torn between two next tokens,
    which is exactly the tight-gap regime where fp precision changes the argmax.
    """
    import random
    rng = random.Random(seed)
    for p in SEED_PROMPTS:
        yield p
    vocab = tok.vocab_size
    for _ in range(pool):
        n = rng.randint(4, 10)
        ids = [rng.randint(0, vocab - 1) for _ in range(n)]
        txt = tok.decode(ids)
        # Skip prompts with quotes/newlines so they survive any plain-text
        # `Text to complete: "..."` framing without truncation.
        if txt.strip() and '"' not in txt and "\n" not in txt and "\r" not in txt:
            yield txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n", type=int, default=64, help="probes to keep")
    ap.add_argument("--pool", type=int, default=600, help="random candidates to mine")
    ap.add_argument("--max-gap", type=float, default=0.05,
                    help="only keep probes whose top-2 logit gap is below this")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "bank.json"))
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    scored = []
    for p in candidate_prompts(tok, args.pool):
        ids = tok(p, return_tensors="pt")
        if ids["input_ids"].shape[1] == 0:
            continue
        with torch.no_grad():
            logits = model(**ids).logits[0, -1].float()
        top = torch.topk(logits, 2)
        gap = (top.values[0] - top.values[1]).item()
        scored.append({
            "prompt": p,
            "gap": gap,
            "argmax_id": int(top.indices[0]),
            "runner_id": int(top.indices[1]),
            "argmax_tok": tok.decode(top.indices[0]),
            "runner_tok": tok.decode(top.indices[1]),
        })

    # Smallest gap first == closest to the decision border == most fragile.
    scored.sort(key=lambda d: d["gap"])
    tight = [d for d in scored if d["gap"] <= args.max_gap]
    keep = tight[: args.n]
    bank = {
        "model": args.model,
        "n": len(keep),
        "max_gap": args.max_gap,
        "pool": args.pool,
        "n_candidates": len(scored),
        "probes": keep,
    }
    with open(args.out, "w") as f:
        json.dump(bank, f, indent=2)
    print(f"scored {len(scored)} candidates; {len(tight)} under gap {args.max_gap}; "
          f"kept {len(keep)} -> {args.out}")
    print("tightest gaps:")
    for d in keep[:10]:
        p = d["prompt"] if len(d["prompt"]) < 48 else d["prompt"][:45] + "..."
        print(f'  gap={d["gap"]:.5f}  {p!r} -> {d["argmax_tok"]!r} / {d["runner_tok"]!r}')


if __name__ == "__main__":
    main()
