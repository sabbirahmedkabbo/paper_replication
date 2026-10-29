#!/usr/bin/env python3
"""
Helper used by replicate_paper.py to read the EIA .xls/.xlsx downloads (read_eia).
Running it directly is optional and only writes CSV copies, which the pipeline does not use:
  brent-daily.csv, brent-weekly.csv, wti-daily.csv, wti-weekly.csv

Works whatever the Excel files are called: the market (Brent / WTI) is read from the
sheet's title text, and daily vs weekly from the spacing of the dates.

  pip install xlrd openpyxl
  python convert_xls.py
"""
import sys
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"


def read_eia(path):
    xl = pd.ExcelFile(path)
    sheet = "Data 1" if "Data 1" in xl.sheet_names else xl.sheet_names[1] if len(xl.sheet_names) > 1 else xl.sheet_names[0]
    raw = xl.parse(sheet, header=None)
    # title text anywhere in the first rows tells us the market
    head = " ".join(str(v) for v in raw.iloc[:4].to_numpy().ravel()).lower()
    if "brent" in head:
        market = "brent"
    elif "wti" in head or "cushing" in head:
        market = "wti"
    else:
        raise ValueError(f"{path.name}: cannot tell Brent from WTI (title: {head[:120]!r})")
    # the header row is the one whose first cell is 'Date'
    hdr = next((i for i in range(min(10, len(raw))) if str(raw.iat[i, 0]).strip().lower() == "date"), None)
    if hdr is None:
        raise ValueError(f"{path.name}: no 'Date' header row found in sheet {sheet!r}")
    df = raw.iloc[hdr + 1:, :2].copy()
    df.columns = ["Date", "Price"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna().sort_values("Date").reset_index(drop=True)
    gap = df["Date"].diff().dt.days.median()
    freq = "weekly" if gap >= 6 else "daily"
    if gap >= 20:
        raise ValueError(f"{path.name}: looks monthly/annual (median gap {gap} days)")
    return market, freq, df


def main():
    files = sorted(p for p in DATA.iterdir() if p.suffix.lower() in (".xls", ".xlsx"))
    if not files:
        sys.exit(f"No .xls/.xlsx files found in {DATA}")
    for p in files:
        try:
            market, freq, df = read_eia(p)
        except Exception as e:
            print(f"SKIP {p.name}: {e}")
            continue
        out = DATA / f"{market}-{freq}.csv"
        df.to_csv(out, index=False)
        print(f"{p.name:<28} -> {out.name:<18} {len(df):>6} rows  "
              f"{df['Date'].iloc[0].date()} -> {df['Date'].iloc[-1].date()}")
    print("\nNow run: python replicate_paper.py check")


if __name__ == "__main__":
    main()
