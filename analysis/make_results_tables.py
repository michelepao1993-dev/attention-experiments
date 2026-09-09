#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


RUN_ORDER = [
    "vanilla",
    "alibi",
    "prope075",
    "s_fixed",
    "m_fixed",
    "mosar0",
    "cost001",
    "rope_m_mask",
]

PRETTY = {
    "vanilla": "RoPE",
    "alibi": "ALiBi",
    "prope075": "ProPE-0.75",
    "s_fixed": "Fixed-S",
    "m_fixed": "Fixed-M",
    "mosar0": "MoSAR0",
    "cost001": "MoSAR-cost",
    "rope_m_mask": "RoPE-M-mask",
}

LENGTHS = [2048, 4096, 8192]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True)
    p.add_argument("--step", type=int, default=610000)
    p.add_argument(
        "--analysis-root",
        default="results/evals",
    )
    p.add_argument("--outdir", default="tables")
    return p.parse_args()


def safe_float(x):
    if x is None or x == "":
        return math.nan
    try:
        return float(x)
    except ValueError:
        return math.nan


def fmt(x, nd=3):
    if x is None or math.isnan(x):
        return "--"
    return f"{x:.{nd}f}"


def fmt_pct(x, nd=1):
    if x is None or math.isnan(x):
        return "--"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.{nd}f}\\%"


def bold_if_best(x, best, nd=3):
    s = fmt(x, nd)
    if not math.isnan(x) and abs(x - best) < 1e-8:
        return "\\textbf{" + s + "}"
    return s


def read_csv(path):
    rows = []
    with open(str(path), newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_all(args):
    data = {}
    for seq in LENGTHS:
        path = Path(args.analysis_root) / f"posttraining_eval_summary_step_{args.step}_seq_{seq}.csv"
        if not path.exists():
            raise SystemExit(f"Missing CSV: {path}")

        rows = read_csv(path)
        data[seq] = rows
        print(f"loaded seq={seq}: {path} rows={len(rows)}")
    return data


def find_row(data, seq, run, source):
    for r in data[seq]:
        if r.get("run") == run and r.get("source") == source and r.get("passed") == "True":
            return r
    return None


def get_soft_row(data, seq, run):
    # At training length, use official main eval.
    # At extrapolation lengths, use post_soft.
    source = "main_soft" if seq == 2048 else "post_soft"
    return find_row(data, seq, run, source)


def get_hard_row(data, seq, run):
    return find_row(data, seq, run, "post_hard_top1")


def write(path, text):
    path.write_text(text)
    print(f"saved={path}")


def make_soft_extrapolation_table(data):
    soft = {}
    for run in RUN_ORDER:
        soft[run] = {}
        for seq in LENGTHS:
            r = get_soft_row(data, seq, run)
            soft[run][seq] = safe_float(r.get("ppl")) if r else math.nan

    best = {}
    for seq in LENGTHS:
        vals = [soft[run][seq] for run in RUN_ORDER if not math.isnan(soft[run][seq])]
        best[seq] = min(vals)

    vanilla_8192 = soft["vanilla"][8192]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Model & PPL@2k & PPL@4k & PPL@8k & $\\Delta$ vs RoPE@8k \\\\")
    lines.append("\\midrule")

    # Sort by PPL@8192, best first; keep invalid at bottom.
    ordered = sorted(
        RUN_ORDER,
        key=lambda r: soft[r][8192] if not math.isnan(soft[r][8192]) else 1e9,
    )

    for run in ordered:
        p2 = soft[run][2048]
        p4 = soft[run][4096]
        p8 = soft[run][8192]
        if math.isnan(p8) or math.isnan(vanilla_8192):
            d8 = math.nan
        else:
            d8 = 100.0 * (p8 - vanilla_8192) / vanilla_8192

        model = PRETTY.get(run, run)
        if run == "mosar0":
            model = "\\textbf{" + model + "}"

        lines.append(
            f"{model} & "
            f"{bold_if_best(p2, best[2048])} & "
            f"{bold_if_best(p4, best[4096])} & "
            f"{bold_if_best(p8, best[8192])} & "
            f"{fmt_pct(d8)} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{")
    lines.append("Length extrapolation in soft/native inference mode. ")
    lines.append("All models are evaluated on the same validation corpus, rebuilt at each context length. ")
    lines.append("Lower perplexity is better. The last column reports relative change with respect to RoPE at 8k tokens.")
    lines.append("}")
    lines.append("\\label{tab:length-extrapolation-soft}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def make_hard_approx_table(data):
    runs = ["mosar0", "cost001"]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3.2pt}")
    lines.append("\\begin{tabular}{llrrrrr}")
    lines.append("\\toprule")
    lines.append("Model & Length & Soft PPL & Hard PPL & Gap & Soft cost & Hard cost \\\\")
    lines.append("\\midrule")

    for run in runs:
        for seq in LENGTHS:
            soft = get_soft_row(data, seq, run)
            hard = get_hard_row(data, seq, run)

            if not soft or not hard:
                continue

            sp = safe_float(soft.get("ppl"))
            hp = safe_float(hard.get("ppl"))
            sc = safe_float(soft.get("cost"))
            hc = safe_float(hard.get("cost"))

            gap_pct = math.nan
            if not math.isnan(sp) and not math.isnan(hp):
                gap_pct = 100.0 * (hp - sp) / sp

            model = PRETTY.get(run, run)
            if run == "mosar0":
                model = "\\textbf{" + model + "}"

            lines.append(
                f"{model} & {seq//1024}k & "
                f"{fmt(sp)} & {fmt(hp)} & {fmt_pct(gap_pct)} & "
                f"{fmt(sc, 4)} & {fmt(hc, 4)} \\\\"
            )
        if run != runs[-1]:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{")
    lines.append("Hard top-1 approximation of learned MoSAR geometries. ")
    lines.append("The gap is the relative perplexity increase of hard routing with respect to the corresponding soft geometry. ")
    lines.append("Hard routing is not intended to improve over soft routing, but to test whether the learned controlled-decay geometry admits a low-cost approximation.")
    lines.append("}")
    lines.append("\\label{tab:hard-approximation}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def make_mosar_routing_table(data):
    runs = ["mosar0", "cost001"]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{tabular}{llrrrrrrr}")
    lines.append("\\toprule")
    lines.append("Model & Length & Cost & $q_S$ & $q_M$ & $q_G$ & $k_S$ & $k_M$ & $k_G$ \\\\")
    lines.append("\\midrule")

    for run in runs:
        for seq in LENGTHS:
            r = get_soft_row(data, seq, run)
            if not r:
                continue

            model = PRETTY.get(run, run)
            if run == "mosar0":
                model = "\\textbf{" + model + "}"

            vals = [
                safe_float(r.get("cost")),
                safe_float(r.get("q_S")),
                safe_float(r.get("q_M")),
                safe_float(r.get("q_G")),
                safe_float(r.get("k_S")),
                safe_float(r.get("k_M")),
                safe_float(r.get("k_G")),
            ]

            lines.append(
                f"{model} & {seq//1024}k & "
                + " & ".join(fmt(v, 3) for v in vals)
                + " \\\\"
            )
        if run != runs[-1]:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{")
    lines.append("Soft MoSAR routing statistics across context lengths. ")
    lines.append("MoSAR0 keeps a stable query-side distribution while gradually shifting key-side mass from short to medium/global regimes as context length increases.")
    lines.append("}")
    lines.append("\\label{tab:mosar-routing}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def make_compact_csv(data, outdir):
    path = outdir / "paper_length_extrapolation_compact.csv"
    fieldnames = [
        "model",
        "ppl_2048_soft",
        "ppl_4096_soft",
        "ppl_8192_soft",
        "cost_2048_soft",
        "cost_4096_soft",
        "cost_8192_soft",
        "ppl_2048_hard",
        "ppl_4096_hard",
        "ppl_8192_hard",
        "cost_2048_hard",
        "cost_4096_hard",
        "cost_8192_hard",
    ]

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for run in RUN_ORDER:
            row = {"model": PRETTY.get(run, run)}
            for seq in LENGTHS:
                soft = get_soft_row(data, seq, run)
                hard = get_hard_row(data, seq, run)

                row[f"ppl_{seq}_soft"] = fmt(safe_float(soft.get("ppl")), 6) if soft else ""
                row[f"cost_{seq}_soft"] = fmt(safe_float(soft.get("cost")), 6) if soft else ""
                row[f"ppl_{seq}_hard"] = fmt(safe_float(hard.get("ppl")), 6) if hard else ""
                row[f"cost_{seq}_hard"] = fmt(safe_float(hard.get("cost")), 6) if hard else ""

            w.writerow(row)

    print(f"saved={path}")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_all(args)

    write(outdir / "table_length_extrapolation_soft.tex", make_soft_extrapolation_table(data))
    write(outdir / "table_hard_approximation.tex", make_hard_approx_table(data))
    write(outdir / "table_mosar_routing.tex", make_mosar_routing_table(data))
    make_compact_csv(data, outdir)


if __name__ == "__main__":
    main()
