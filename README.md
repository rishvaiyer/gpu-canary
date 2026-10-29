# GPU Canary

A language model can check the arithmetic of the GPU it runs on. Mine 64
prompts where the model's top two next-token logits are nearly tied (border
inputs), record the greedy answers while the hardware is healthy, replay them
later, count mismatches. One scan is 64 single-token forwards: 2.5 s for a
0.5B model on an NVIDIA L4 in fp16.

Paper: `paper/main.tex` (9 pages). Author: Rishva Tejan Iyer, independent researcher.

## What it finds

Against software-injected faults that follow the patterns reported in
production SDC studies (one element zeroed, sign flipped, bit flipped, or
biased by 1e-3), on an L4 in fp16, Qwen2.5-0.5B:

| fault, rate 0.01 per Linear forward | border probes | ordinary prompts |
|---|---|---|
| element zeroed | 10 / 10 scans | 0 / 10 |
| sign flipped | 10 / 10 | 0 / 10 |
| random bit flipped | 10 / 10 | 8 / 10 |
| bias 1e-3 | 10 / 10 | 0 / 10 |

For a unit that is wrong the same way on every forward (a persistent fault),
redundant execution (run twice, compare) flags 0 of 32 sites because both runs
carry the same error. Border probes flag 32 of 32 on two models. Ordinary
prompts flag 1 and 0.

## What it does not show

- No genuinely defective GPU has been tested. Every fault is injected in software.
- Models are small (Qwen2.5-0.5B and 1.5B), scans are unbatched in the
  HuggingFace path. Serving engines with dynamic batching are not batch-invariant;
  the false-alarm rate there is unmeasured.
- Ten scans per cell, eight sites per pattern. Wide confidence intervals.

The paper's Limitations section is the part to read first.

## Layout

```
probes/make_probes.py   mine a border-probe bank for a model (fp32, gap < 0.05)
probes/bank.json        the 64-probe bank for Qwen2.5-0.5B used in the paper
refs/run_ref.py         record a reference signature for one stack
refs/sig_*.json         references: L4 and Ryzen 7 2700, fp32 / fp16 / bf16
health/check.py         replay the bank, count mismatches; --inject for lane faults
health/faultsim.py      intermittent element faults (zero, sign, bitflip, bias)
health/persist.py       persistent element faults vs redundant execution
health/prism_fault.py   LLM-PRISM RTL-signature defects (tensor core, FMA, L1)
data/                   every run log behind every number in the paper
paper/                  LaTeX source, bibliography, figures, make_figures.py
```

## Run

```
pip install -r requirements.txt
python probes/make_probes.py --model Qwen/Qwen2.5-0.5B            # or use probes/bank.json
python refs/run_ref.py --label mine --dtype float16 --device cuda --out refs/sig_mine.json
python health/check.py --ref refs/sig_mine.json --dtype float16 --device cuda
python health/faultsim.py --fault zero --rate 0.01 --reps 10 --dtype float16 --device cuda --log data/my_faultsim.jsonl
python paper/make_figures.py
```

Each script prints its own `--help`. Model weights download from HuggingFace on
first run.

## Prior work this builds on

Border inputs come from HSPI (Zhang et al., arXiv:2411.05197) and B3IT
(Chauvin et al., ICML 2026, arXiv:2602.11083), which use them to identify a
serving platform or detect a changed model behind an API. This repository
points the same probes at hardware integrity. Fault patterns follow SDCHunter
and AEGIS (OSDI 2026), Tung et al. (arXiv:2605.04213) and LLM-PRISM
(arXiv:2604.10390). Full references in `paper/refs.bib`.

## License

Code: MIT. Paper and figures: CC BY 4.0.
