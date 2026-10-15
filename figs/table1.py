"""Table I — out-of-sample coverage and capacity cost per lead band (test period, Drake / 75 degC) for the conformal bound
and the comparison margins. Data: results/Q2Q3_test.csv. Writes paper/tables/table1.tex. Regenerate: python figs/table1.py"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
q = pd.read_csv(REPO / "results" / "Q2Q3_test.csv")
q = q[q.subset == "all"]
BANDS = ["1-6", "7-24", "25-48", "49-96"]
METHODS = [("AARGUARD", "Conformal (this paper)"), ("AARGUARD-ACI", "Conformal, adaptive"), ("B2", "Fixed margin"),
           ("B3", "Gaussian, fitted"), ("B3n", "Gaussian, nominal"), ("B3r-n", "Gaussian, rolling")]


def cell(method, alpha, col, band):
    r = q[(q.method == method) & (q.alpha.round(3) == alpha) & (q.band == band)]
    return 100 * float(r[col].iloc[0])


def covcell(method, alpha, band):
    """Coverage in %, bold when below the nominal 1 - alpha (unrounded comparison)."""
    v = cell(method, alpha, "cov", band)
    return rf"\textbf{{{v:.1f}}}" if v < 100 * (1 - alpha) - 1e-9 else f"{v:.1f}"


lines = [r"\begin{table*}[!t]", r"\footnotesize", r"\caption{Coverage and capacity cost by lead band, test period, Drake ACSR at $75\,^\circ$C. Bold coverage is below $1-\alpha$}",
         r"\label{tab:coverage}", r"\centering", r"\setlength{\tabcolsep}{2.6pt}",
         r"\begin{tabular}{l cccc c cccc c cccc c cccc}", r"\toprule",
         r" & \multicolumn{4}{c}{Coverage (\%), $\alpha=1\%$} & & \multicolumn{4}{c}{CC (\%), $\alpha=1\%$} & & \multicolumn{4}{c}{Coverage (\%), $\alpha=5\%$} & & \multicolumn{4}{c}{CC (\%), $\alpha=5\%$} \\",
         r"\cmidrule{2-5}\cmidrule{7-10}\cmidrule{12-15}\cmidrule{17-20}",
         r"Lead band (h) & 6 & 7--24 & 25--48 & 49--72 & & 6 & 7--24 & 25--48 & 49--72 & & 6 & 7--24 & 25--48 & 49--72 & & 6 & 7--24 & 25--48 & 49--72 \\", r"\midrule"]
for m, name in METHODS:
    c1 = [covcell(m, 0.01, b) for b in BANDS]
    c5 = [covcell(m, 0.05, b) for b in BANDS]
    k1 = [f"{cell(m, 0.01, 'CC', b):.1f}" for b in BANDS]
    cc = [f"{cell(m, 0.05, 'CC', b):.1f}" for b in BANDS]
    lines.append(f"{name} & " + " & ".join(c1) + " & & " + " & ".join(k1) + " & & " + " & ".join(c5) + " & & " + " & ".join(cc) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
out = REPO / "paper" / "tables" / "table1.tex"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
