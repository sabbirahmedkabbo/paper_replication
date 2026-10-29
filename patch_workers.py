#!/usr/bin/env python3
"""Bypass multiprocessing.Pool when --workers 1 to fix macOS arm64 TF deadlock."""
from pathlib import Path

SRC = Path("replicate_paper.py")
text = SRC.read_text()
before = len(text)

OLD = '''    if not jobs:
        return
    ctx = mp.get_context("spawn")
    t0, done = time.time(), 0
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:'''

NEW = '''    if not jobs:
        return
    if args.workers == 1:
        # WORKER-FIX: macOS arm64 TF deadlocks inside multiprocessing.Pool workers.
        t0, done = time.time(), 0
        for job in jobs:
            job_r, sec, status = run_job(job)
            done += 1
            el = time.time() - t0
            eta = el / done * (len(jobs) - done)
            ds, mname, variant, run, _ = job_r
            print(f"[{done}/{len(jobs)}] {ds} {mname} {variant} run{run:02d}: {status.splitlines()[0]} "
                  f"({sec / 60:.1f} min) | elapsed {el / 3600:.2f} h, ETA {eta / 3600:.2f} h")
            if status.startswith("ERROR"):
                print(status)
        return
    ctx = mp.get_context("spawn")
    t0, done = time.time(), 0
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:'''

assert OLD in text, "FIND block not found - wrong file or already patched"
assert text.count(OLD) == 1, f"FIND block appears {text.count(OLD)} times"

text = text.replace(OLD, NEW, 1)
SRC.write_text(text)
print(f"OK: patched replicate_paper.py ({before} -> {len(text)} bytes)")