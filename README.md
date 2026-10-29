# Replication of Shen et al. (2024)

Python replication of the paper "Intelligent crude oil price probability forecasting:
Deep learning models and industry applications" (Computers in Industry, 2024).

Result: CNN-BiLSTM baseline reproduces the paper (RMSE 2.83 vs 2.68).
EEMD-augmented models diverge - see replication_report.docx for analysis.

## Reproduce

    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python replicate_paper.py all --datasets wti_weekly --runs 1 --workers 1 --threads 2
