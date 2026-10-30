#!/usr/bin/env python3
"""
census/run_census.py -- Rent many distinct GPUs on Vast.ai, scan each once,
collect results, destroy the instance. Budget-capped.

    pip install vastai
    vastai set api-key <KEY>
    python census/run_census.py --max-machines 200 --max-usd 20 --max-dph 0.40 --out data/census.jsonl

What it does, per machine:
  1. picks an offer (one per distinct machine_id, cheapest first, verified hosts
     by default) that matches the filters,
  2. creates an instance on a pytorch image with an onstart command that fetches
     census/scan_node.py from the public repo and runs it,
  3. polls `vastai logs` for the CENSUS_RESULT line (timeout per machine),
  4. appends the result to --out and destroys the instance.

Spend is estimated as dph * wall-clock and stops new launches once --max-usd
is projected to be exceeded. Nothing here is free: this script RENTS hardware.
Run with --dry-run first to see the offers it would take.

UNTESTED AGAINST THE LIVE API from the machine where it was written (no
network). Flag names follow the vastai CLI as of 2026; check `vastai --help`
if a call errors and adjust the three subprocess wrappers at the top.
"""
import argparse, json, subprocess, sys, time

ONSTART = ("cd /root && (curl -fsSL {raw}/census/scan_node.py -o scan_node.py || "
           "wget -q {raw}/census/scan_node.py -O scan_node.py) && "
           "pip install -q transformers 2>/dev/null; "
           "CANARY_REPEATS={reps} python3 scan_node.py; sleep 3600")

def vast(*args, raw=True):
    cmd = ["vastai", *args] + (["--raw"] if raw else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {p.stderr.strip()[:400]}")
    return json.loads(p.stdout) if raw else p.stdout

def search(a):
    q = (f"gpu_ram>={a.min_gpu_ram} dph<={a.max_dph} rentable=true num_gpus=1 "
         f"inet_down>=100 disk_space>={a.disk} cuda_vers>={a.min_cuda}")
    if a.verified: q += " verified=true"
    if a.gpu_name: q += f" gpu_name={a.gpu_name}"
    offers = vast("search", "offers", q, "-o", "dph")
    seen, picks = set(), []
    for o in offers:
        m = o.get("machine_id")
        if m in seen: continue
        seen.add(m); picks.append(o)
    return picks

def launch(offer, a):
    r = vast("create", "instance", str(offer["id"]), "--image", a.image,
             "--disk", str(a.disk), "--label", "gpu-canary-census",
             "--onstart-cmd", ONSTART.format(raw=a.raw, reps=a.repeats))
    return r["new_contract"]

def wait_result(iid, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            log = vast("logs", str(iid), raw=False)
        except RuntimeError:
            log = ""
        for line in log.splitlines():
            if line.startswith("CENSUS_RESULT "):
                return json.loads(line[len("CENSUS_RESULT "):])
        time.sleep(20)
    return None

def destroy(iid):
    try: vast("destroy", "instance", str(iid), raw=False)
    except RuntimeError as e: print(f"  destroy {iid} failed: {e}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-machines", type=int, default=50)
    ap.add_argument("--max-usd", type=float, default=10.0)
    ap.add_argument("--max-dph", type=float, default=0.40)
    ap.add_argument("--min-gpu-ram", type=int, default=8)
    ap.add_argument("--min-cuda", type=float, default=12.1)
    ap.add_argument("--disk", type=int, default=16)
    ap.add_argument("--gpu-name", default=None, help='e.g. "RTX_3090"')
    ap.add_argument("--verified", action="store_true", default=True)
    ap.add_argument("--unverified", dest="verified", action="store_false")
    ap.add_argument("--image", default="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    ap.add_argument("--raw", default="https://raw.githubusercontent.com/rishvaiyer/gpu-canary/main")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per machine")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--out", default="data/census.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    offers = search(a)[: a.max_machines]
    print(f"{len(offers)} distinct machines match; cheapest {offers[0]['dph_total']:.3f} $/h" if offers else "no offers")
    if a.dry_run:
        for o in offers:
            print(f"  offer {o['id']:>9} machine {o['machine_id']:>7} {o.get('gpu_name','?'):<18} "
                  f"{o.get('dph_total',0):.3f} $/h  rel {o.get('reliability2',0):.3f}  {o.get('geolocation','')}")
        return

    spent, active, queue, done = 0.0, {}, list(offers), 0
    while queue or active:
        while queue and len(active) < a.parallel:
            o = queue.pop(0)
            est = o["dph_total"] * a.timeout / 3600
            if spent + est > a.max_usd:
                print(f"budget: skipping remaining {len(queue)+1} offers"); queue.clear(); break
            try:
                iid = launch(o, a)
                active[iid] = (o, time.time())
                print(f"launched {iid} on machine {o['machine_id']} {o.get('gpu_name')}")
            except RuntimeError as e:
                print(f"launch failed for offer {o['id']}: {e}", file=sys.stderr)
        for iid in list(active):
            o, t0 = active[iid]
            res = wait_result(iid, 20)  # one short poll
            if res is None and time.time() - t0 < a.timeout: continue
            hours = (time.time() - t0) / 3600
            spent += hours * o["dph_total"]
            rec = {"offer": {k: o.get(k) for k in ("id", "machine_id", "gpu_name", "dph_total",
                                                     "reliability2", "geolocation", "host_id", "verified")},
                   "instance": iid, "wall_hours": round(hours, 3), "result": res}
            with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
            destroy(iid); del active[iid]; done += 1
            st = res["status"] if res else "TIMEOUT"
            print(f"[{done}] machine {o['machine_id']} {o.get('gpu_name')}: {st}  (spent ${spent:.2f})")
    print(f"done: {done} machines, ${spent:.2f} estimated")

if __name__ == "__main__":
    main()
