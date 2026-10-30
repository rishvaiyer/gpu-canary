#!/usr/bin/env python3
"""
census/scan_node.py -- One-file GPU Canary scan for a machine you do not own.

Runs on any rented GPU box with Python 3.9+ and internet. Installs nothing
beyond torch + transformers (already present on the pytorch/pytorch images).
Downloads the probe bank and the fp32 reference signature from the public repo,
runs the 64 border probes in fp32, fp16 and bf16 (REPEATS times each), and
prints one line:

    CENSUS_RESULT {json}

The orchestrator (run_census.py) reads that line from the instance log.

Why fp32 is the universal reference: in the paper's data the fp32 signature
is identical on an NVIDIA L4 (CUDA) and on a Ryzen 7 2700 (CPU), 0 of 64
differ. So on an unknown card, any fp32 mismatch is a finding. fp16 and bf16
signatures are stack-specific (L4 vs Ryzen differ by 4 and 20), so those are
judged by peer majority in analyze.py, not against a fixed reference.

Exit code 0 = fp32 clean, 2 = fp32 drift, 3 = could not run.
"""
import json, os, platform, subprocess, sys, time, urllib.request

RAW = os.environ.get("CANARY_RAW",
      "https://raw.githubusercontent.com/rishvaiyer/gpu-canary/main")
REPEATS = int(os.environ.get("CANARY_REPEATS", "2"))
DTYPES = os.environ.get("CANARY_DTYPES", "float32,float16,bfloat16").split(",")
OUT = os.environ.get("CANARY_OUT", "/root/census_result.json")

def fetch_json(path):
    with urllib.request.urlopen(f"{RAW}/{path}", timeout=60) as r:
        return json.load(r)

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as e:
        return f"ERR {e}"

def hw_info():
    q = "name,driver_version,memory.total,pci.bus_id,serial,uuid,temperature.gpu,clocks.sm,clocks.mem,power.draw,ecc.errors.uncorrected.aggregate.total"
    smi = sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader")
    return {
        "nvidia_smi": smi,
        "gpu_name": smi.split(",")[0].strip() if smi and not smi.startswith("ERR") else None,
        "host": platform.node(), "machine": platform.machine(),
        "cpu": sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"),
        "python": sys.version.split()[0],
        "vast_instance": os.environ.get("VAST_CONTAINERLABEL") or os.environ.get("CONTAINER_ID"),
        "public_ip": sh("curl -s -m 5 https://api.ipify.org || true"),
    }

def main():
    t_start = time.time()
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hw": hw_info(),
           "repeats": REPEATS, "scans": {}, "status": "ERROR"}
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        rec["torch"] = torch.__version__
        rec["cuda"] = torch.version.cuda
        rec["cudnn"] = torch.backends.cudnn.version()
        rec["gpu_torch"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        rec["capability"] = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        device = "cuda" if torch.cuda.is_available() else "cpu"

        bank = fetch_json("probes/bank.json")
        ref32 = fetch_json("refs/sig_l4_float32.json")["signature"]
        prompts = [p["prompt"] for p in bank["probes"]]
        rec["model"] = bank["model"]
        tok = AutoTokenizer.from_pretrained(bank["model"])
        enc = [tok(p, return_tensors="pt") for p in prompts]

        for dname in DTYPES:
            dt = getattr(torch, dname)
            t0 = time.time()
            model = AutoModelForCausalLM.from_pretrained(bank["model"], dtype=dt).to(device).eval()
            t_load = time.time() - t0
            runs, gaps = [], []
            for _ in range(REPEATS):
                sig, gap = [], []
                t1 = time.time()
                with torch.no_grad():
                    for ids in enc:
                        logits = model(**{k: v.to(device) for k, v in ids.items()}).logits[0, -1].float().cpu()
                        top = torch.topk(logits, 2)
                        sig.append(int(torch.argmax(logits)))
                        gap.append(round((top.values[0] - top.values[1]).item(), 6))
                runs.append(sig); gaps.append(gap)
                rec["scans"].setdefault(dname, {})["seconds_per_scan"] = round(time.time() - t1, 2)
            s = rec["scans"][dname]
            s["load_seconds"] = round(t_load, 1)
            s["signatures"] = runs
            s["top2_gap_first"] = gaps[0]
            s["repeat_disagree"] = sum(any(r[i] != runs[0][i] for r in runs[1:]) for i in range(len(runs[0])))
            if dname == "float32":
                s["mismatch_vs_universal_fp32"] = [i for i, (a, b) in enumerate(zip(runs[0], ref32)) if a != b]
            del model
            if device == "cuda": torch.cuda.empty_cache()

        f32 = rec["scans"].get("float32", {})
        bad = len(f32.get("mismatch_vs_universal_fp32", [])) + f32.get("repeat_disagree", 0)
        rec["status"] = "DRIFT" if bad else "OK"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["total_seconds"] = round(time.time() - t_start, 1)
    line = json.dumps(rec, separators=(",", ":"))
    try:
        with open(OUT, "w") as f: f.write(line)
    except Exception:
        pass
    print("CENSUS_RESULT " + line, flush=True)
    sys.exit({"OK": 0, "DRIFT": 2}.get(rec["status"], 3))

if __name__ == "__main__":
    main()
