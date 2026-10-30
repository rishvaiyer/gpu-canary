#!/usr/bin/env python3
"""
census/analyze.py -- Turn data/census.jsonl into the table the paper needs.

Three signals per machine:
  fp32_mismatch   : fp32 signature differs from the universal fp32 reference
                    (identical on L4 and Ryzen in the paper). Any nonzero = finding.
  repeat_disagree : the same dtype scanned twice on the same card gave different
                    argmax tokens. Intermittent fault or nondeterministic kernel.
  peer_outlier    : fp16/bf16 signature differs from the majority signature of
                    other machines with the same (gpu_name, torch, cuda). Persistent
                    fault or a different kernel path on that one box.

    python census/analyze.py data/census.jsonl
"""
import collections, json, sys

def main(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    ok = [r for r in rows if r.get("result") and r["result"].get("status") in ("OK", "DRIFT")]
    print(f"{len(rows)} machines rented, {len(ok)} scanned, {len(rows)-len(ok)} failed/timeout")

    groups = collections.defaultdict(list)
    for r in ok:
        res = r["result"]
        key = (res.get("gpu_torch") or res["hw"].get("gpu_name"), res.get("torch"), res.get("cuda"))
        groups[key].append(r)

    findings = []
    for key, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        for d in ("float16", "bfloat16"):
            sigs = [tuple(r["result"]["scans"][d]["signatures"][0]) for r in rs if d in r["result"]["scans"]]
            if not sigs: continue
            mode, n_mode = collections.Counter(sigs).most_common(1)[0]
            for r in rs:
                s = r["result"]["scans"].get(d)
                if not s: continue
                diff = sum(a != b for a, b in zip(s["signatures"][0], mode))
                if diff and n_mode >= 2:
                    findings.append((r, f"peer_outlier {d}: {diff}/64 differ from majority of {len(sigs)} peers"))
        for r in rs:
            f32 = r["result"]["scans"].get("float32", {})
            mm = f32.get("mismatch_vs_universal_fp32", [])
            if mm: findings.append((r, f"fp32_mismatch: probes {mm}"))
            for d, s in r["result"]["scans"].items():
                if s.get("repeat_disagree"): findings.append((r, f"repeat_disagree {d}: {s['repeat_disagree']}/64"))

    print("\nGPU population:")
    for key, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(rs):>4}  {key[0]}  torch {key[1]}  cuda {key[2]}")
    print(f"\nFindings: {len(findings)}")
    for r, why in findings:
        o = r["offer"]
        print(f"  machine {o['machine_id']} host {o.get('host_id')} {o.get('gpu_name')} {o.get('geolocation','')}: {why}")
    secs = [r["result"]["scans"]["float16"]["seconds_per_scan"] for r in ok if "float16" in r["result"]["scans"]]
    if secs: print(f"\nfp16 scan time: median {sorted(secs)[len(secs)//2]} s over {len(secs)} cards")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/census.jsonl")
