#!/usr/bin/env python3
"""Apply the 4 bug fixes + 3 warnings to replicate_paper.py.
Verifies each replacement succeeded. Fails loudly if any is ambiguous."""
import re
from pathlib import Path

SRC = Path("replicate_paper.py")
text = SRC.read_text()
original = text

fixes = []  # (name, old, new)

# ---- Fix 1: add hashlib import ----
fixes.append(("import hashlib",
    "import numpy as np\nimport pandas as pd",
    "import hashlib\nimport numpy as np\nimport pandas as pd"))

# ---- Fix 2: add CFG_HASH constant after CHOICES block ----
fixes.append(("CFG_HASH constant",
    ' 10. Adam with Keras defaults (learning rate 1e-3); no early stopping, no validation split.\n"""\n',
    ' 10. Adam with Keras defaults (learning rate 1e-3); no early stopping, no validation split.\n"""\n\n'
    'CFG_HASH = hashlib.md5(\n'
    '    (CHOICES + f"|LAG={LAG}|EPOCHS={EPOCHS}|BATCH={BATCH}|DROPOUT={DROPOUT}"\n'
    '     f"|QR={QR_MODE}|CNN2LSTM={CNN_TO_LSTM}|DELTA={DELTA_TARGETS}"\n'
    '     f"|EEMD={EEMD_TRIALS},{EEMD_NSTD},{EEMD_SEED}").encode()\n'
    ').hexdigest()[:8]\n'))

# ---- Fix 3: align() pads with zeros ----
fixes.append(("align() zero-pad",
    '''    pad = np.zeros((n - m, comps.shape[1]))
    return np.vstack([comps[:-1], pad, comps[-1:]])''',
    '''    pad_val = float(comps[-1].mean())
    pad = np.full((n - m, comps.shape[1]), pad_val)
    return np.vstack([comps[:-1], pad, comps[-1:]])'''))

# ---- Fix 4: pinaw() wrong denominator ----
fixes.append(("pinaw signature",
    '''def pinaw(y, q):
    return float(np.mean(q[:, -1] - q[:, 0]) / (y.max() - y.min()))''',
    '''def pinaw(y, q, scale):
    return float(np.mean(q[:, -1] - q[:, 0]) / scale)'''))

fixes.append(("pinaw call site",
    '''             PINAW=pinaw(y, pred), PICP=picp(y, pred))''',
    '''             PINAW=pinaw(y, pred, P["scale"]), PICP=picp(y, pred))'''))

# ---- Fix 5: crps_kde truncation ----
fixes.append(("crps_kde grid",
    '''def crps_kde(y, q, grid_n=800):''',
    '''def crps_kde(y, q, grid_n=2000):'''))

fixes.append(("crps_kde bounds",
    '''        a = min(s.min(), y[t]) - 5 * h
        b = max(s.max(), y[t]) + 5 * h''',
    '''        a = min(s.min(), y[t]) - 20 * h
        b = max(s.max(), y[t]) + 20 * h'''))

# ---- Fix 6: pred_path uses CFG_HASH ----
fixes.append(("pred_path hash",
    '''    return p / f"{model}_{variant}_run{run:02d}.npz"''',
    '''    return p / f"{model}_{variant}_run{run:02d}_{CFG_HASH}.npz"'''))

# ---- Fix 7: metrics_for uses CFG_HASH ----
fixes.append(("metrics_for hash",
    '''    cp = cache / f"{mname}_{variant}_run{run:02d}.json"''',
    '''    cp = cache / f"{mname}_{variant}_run{run:02d}_{CFG_HASH}.json"'''))

# ---- Fix 8: warn on n != paper_n ----
fixes.append(("prepare warning",
    '''    split = c["n_train"] if n == c["n"] else int(n * 2 / 3)''',
    '''    if n != c["n"]:
        print(f"  WARNING [{ds}]: n={n} != paper_n={c['n']} "
              f"(short by {c['n'] - n}); using int(2/3 * n) split")
        split = int(n * 2 / 3)
    else:
        split = c["n_train"]'''))

# ---- Fix 9: assert decomposition reconstructs ----
fixes.append(("decompose assert",
    '''        print(f"  components: {n_comp}; max |signal - sum| = {gap:.2e}")''',
    '''        print(f"  components: {n_comp}; max |signal - sum| = {gap:.2e}")
        assert gap < 1e-8, f"Decomposition does not reconstruct: gap={gap}"'''))

# ---- Fix 10: warn on ragged runs ----
fixes.append(("ragged runs warning",
    '''            L = min(len(tab[m]) for m in names)
            M = np.array([[tab[m][r][key] for m in names] for r in range(L)])''',
    '''            Ls = {m: len(tab[m]) for m in names}
            if len(set(Ls.values())) > 1:
                print(f"  WARNING: ragged runs {Ls}; using L={min(Ls.values())} "
                      f"(results below are PRELIMINARY)")
            L = min(Ls.values())
            M = np.array([[tab[m][r][key] for m in names] for r in range(L)])'''))

# ---- Apply ----
for name, old, new in fixes:
    if old not in text:
        print(f"FAIL  {name!r}: pattern not found")
        raise SystemExit(1)
    if text.count(old) > 1:
        print(f"FAIL  {name!r}: pattern appears {text.count(old)} times")
        raise SystemExit(1)
    text = text.replace(old, new, 1)
    print(f"OK    {name}")

if text == original:
    print("No changes made.")
    raise SystemExit(0)

SRC.write_text(text)
print(f"\nWrote {SRC}  (sha256 prefix: {hashlib.sha256(text.encode()).hexdigest()[:12]})")
print("\nVerify with:  python -c \"import ast; ast.parse(open('replicate_paper.py').read()); print('syntax OK')\"")