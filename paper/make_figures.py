#!/usr/bin/env python3
"""Figures for the GPU Canary paper. Every number is read from the run logs in ../data."""
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "pdf.fonttype": 42, "ps.fonttype": 42})
BORDER, ORD = "#15212D", "#9DB4CC"

def rows(name):
    return [json.loads(l) for l in open(os.path.join(DATA, name)) if l.strip().startswith("{")]

# Figure 1: scan-level recall, L4 fp16, intermittent faults
r = rows("l4_faultsim.jsonl")
faults = ["zero", "sign", "bitflip", "bias"]; rates = [1.0, 0.1, 0.01]
labels = {"zero": "element zeroed", "sign": "sign flipped", "bitflip": "random bit flipped", "bias": "bias 1e-3"}
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5), sharey=True)
for ax, f in zip(axes, faults):
    b = [next(x["border_recall"] for x in r if x["fault"] == f and x["rate"] == rt) for rt in rates]
    o = [next(x["ordinary_recall"] for x in r if x["fault"] == f and x["rate"] == rt) for rt in rates]
    xs = range(len(rates)); w = 0.38
    ax.bar([x - w/2 for x in xs], b, w, color=BORDER, label="border probes (64)")
    ax.bar([x + w/2 for x in xs], o, w, color=ORD, label="ordinary prompts (32)")
    ax.set_xticks(list(xs)); ax.set_xticklabels([f"{rt:g}" for rt in rates]); ax.set_ylim(0, 1.05)
    ax.set_title(labels[f], fontsize=9)
axes[0].set_ylabel("fraction of scans flagged (of 10)")
fig.supxlabel("corruption rate per Linear forward", fontsize=8.5)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2)
fig.subplots_adjust(bottom=0.22)
fig.savefig(os.path.join(FIG, "fig_recall_l4.pdf")); plt.close(fig)

# Figure 2: lane-fault sensitivity, L4 fp16 vs Ryzen fp32
def lane(name):
    out = {"border": {}, "ordinary": {}}
    for x in rows(name):
        if not x["inject"]: continue
        eps = float(x["inject"].split(":")[2]); out[x["set"]][eps] = x["mismatches"] / x["n"]
    return out
l4 = lane("l4_sweep.jsonl")
ry = lane("health_sweep.jsonl"); ry2 = lane("health_sweep2.jsonl")
for k in ry: ry[k].update(ry2[k])
fig, ax = plt.subplots(figsize=(3.5, 2.4))
for d, name, ls in [(l4, "L4 fp16", "-"), (ry, "Ryzen fp32", "--")]:
    for s, col, mk in [("border", BORDER, "o"), ("ordinary", ORD, "s")]:
        e = sorted(k for k in d[s] if k > 0)
        ax.plot(e, [d[s][k] for k in e], ls, color=col, marker=mk, ms=3.5, lw=1.2, label=f"{name}, {s}")
ax.set_xscale("log"); ax.set_xlabel("single lane scale error (1+EPS), EPS"); ax.set_ylabel("fraction of prompts flipped")
ax.legend(frameon=False, fontsize=6.5); ax.set_ylim(-0.02, 0.85)
fig.savefig(os.path.join(FIG, "fig_lane.pdf")); plt.close(fig)

# Figure 3: scans to 90% detection vs fault rate, from measured per-corruption flip probability
q = {}
for x in r:
    if x["rate"] == 0.01: q[x["fault"]] = x["border_probe_flips"] / x["corruptions"]
F = 64 * 168
fig, ax = plt.subplots(figsize=(3.5, 2.4))
rates_x = [10 ** (-k / 4) for k in range(8, 37)]  # 1e-2 .. 1e-9
for f, col, ls in [("sign", BORDER, "-"), ("zero", BORDER, "--"), ("bias", "#CE3F3F", "-")]:
    ys = []
    for rt in rates_x:
        p = 1 - (1 - q[f]) ** (F * rt); ys.append(math.log(0.1) / math.log(1 - p) if 0 < p < 1 else 1)
    ax.plot(rates_x, ys, ls, color=col, lw=1.3, label=labels[f] + f" (q={q[f]:.3f})")
ax.axhline(24, color="#73818F", lw=0.8, ls=":"); ax.text(3e-9, 30, "one day of hourly scans", fontsize=6.5, color="#73818F", ha="right", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
ax.axhline(1440 * 24, color="#73818F", lw=0.8, ls=":"); ax.text(3e-9, 1440 * 24 * 1.4, "one day of continuous 2.5 s scans", fontsize=6.5, color="#73818F", ha="right", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
ax.set_xlabel("fault rate per Linear forward"); ax.set_ylabel("scans to 90% detection")
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
fig.savefig(os.path.join(FIG, "fig_budget.pdf")); plt.close(fig)
print("q per corruption:", {k: round(v, 4) for k, v in q.items()})
print("figures written to", FIG)
