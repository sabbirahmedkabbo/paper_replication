#!/usr/bin/env python3
"""
Pipeline progress monitor for Shen et al. (2024) replication.
Scans results/ for completed .npz prediction files and .json metrics.
"""
from pathlib import Path
import json

OUT = Path("results")

DATASETS = ["brent_daily", "wti_daily", "brent_weekly", "wti_weekly"]
MODELS = ["EEMD-SVR", "EEMD-BiLSTM", "EEMD-CNN", "CNN-BiLSTM", "EEMD-CNN-BiLSTM"]
VARIANTS = ["point", "qr"]
N_RUNS = 20

def main():
    if not OUT.exists():
        print("No 'results/' directory found yet. Run some jobs first.")
        return

    total_expected_nn = len(DATASETS) * len(MODELS) * len(VARIANTS) * N_RUNS
    # SVR is deterministic (run 0 only, but counted as 20 in evaluation)
    svr_models = [m for m in MODELS if "SVR" in m]
    nn_models = [m for m in MODELS if "SVR" not in m]
    
    total_expected = len(DATASETS) * len(VARIANTS) * (len(svr_models) * 1 + len(nn_models) * N_RUNS)
    
    found_preds = 0
    found_metrics = 0
    
    print("=" * 70)
    print(f"{'Dataset':<15} | {'Model':<16} | {'Variant':<8} | {'Completed Runs':<15}")
    print("-" * 70)

    for ds in DATASETS:
        ds_dir = OUT / ds / "preds"
        metrics_dir = OUT / ds / "metrics"
        
        for mname in MODELS:
            is_svr = ("SVR" in mname)
            max_runs = 1 if is_svr else N_RUNS
            
            for variant in VARIANTS:
                completed = 0
                for r in range(max_runs):
                    # Find matching files containing this run configuration
                    pattern = f"{mname}_{variant}_run{r:02d}_*.npz"
                    if list(ds_dir.glob(pattern)):
                        completed += 1
                        found_preds += 1
                
                # Check metrics count if any
                if ds_dir.exists():
                    for p in ds_dir.glob(f"{mname}_{variant}_run*_*.npz"):
                        if list(metrics_dir.glob(p.name.replace(".npz", ".json"))):
                            found_metrics += 1

                status_str = f"{completed}/{max_runs}"
                if is_svr:
                    status_str += " (SVR)"
                print(f"{ds:<15} | {mname:<16} | {variant:<8} | {status_str:<15}")

    print("=" * 70)
    pct = (found_preds / total_expected) * 100
    print(f"Total training prediction files found: {found_preds} / {total_expected} ({pct:.1f}%)")
    print(f"Total compiled metrics files found:   {found_metrics}")
    print("=" * 70)

if __name__ == "__main__":
    main()