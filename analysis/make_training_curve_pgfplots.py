#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


DEFAULT_RUNS = [
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

COLORS = {
    "vanilla": "black",
    "alibi": "blue",
    "prope075": "magenta",
    "s_fixed": "orange",
    "m_fixed": "green!50!black",
    "mosar0": "red",
    "cost001": "cyan!60!black",
    "rope_m_mask": "gray",
}

STYLES = {
    "vanilla": "solid",
    "alibi": "solid",
    "prope075": "dash dot",
    "s_fixed": "dashed",
    "m_fixed": "dashed",
    "mosar0": "very thick, solid",
    "cost001": "very thick, dotted",
    "rope_m_mask": "dash dot",
}

MARKS = {
    "vanilla": "*",
    "alibi": "square*",
    "prope075": "triangle*",
    "s_fixed": "diamond*",
    "m_fixed": "pentagon*",
    "mosar0": "*",
    "cost001": "otimes*",
    "rope_m_mask": "x",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="results/training_curve/training_curve_points_seq2048.csv",
        help="CSV containing one row per run/checkpoint.",
    )
    p.add_argument(
        "--metric",
        choices=["ppl", "lm_loss"],
        default="ppl",
    )
    p.add_argument(
        "--runs",
        nargs="+",
        default=DEFAULT_RUNS,
    )
    p.add_argument(
        "--zoom-from-step",
        type=int,
        default=0,
        help="If set, only plot checkpoints with step >= this value.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output .tex path. If omitted, a default path under figures/ is used.",
    )
    p.add_argument(
        "--height",
        default="0.56\\linewidth",
    )
    return p.parse_args()


def safe_float(x):
    if x is None or x == "":
        return math.nan
    try:
        return float(x)
    except ValueError:
        return math.nan


def load_rows(path, metric, runs, zoom_from_step):
    rows = []
    with open(str(path), newline="") as f:
        for r in csv.DictReader(f):
            run = r.get("run", "")
            if run not in runs:
                continue

            step = int(float(r["step"]))
            if zoom_from_step and step < zoom_from_step:
                continue

            value = safe_float(r.get(metric))
            if math.isnan(value):
                continue

            step_k = safe_float(r.get("step_k"))
            if math.isnan(step_k):
                step_k = step / 1000.0

            rows.append(
                {
                    "run": run,
                    "step": step,
                    "step_k": step_k,
                    "value": value,
                }
            )

    if not rows:
        raise SystemExit("No usable rows found. Check input CSV, metric, and run names.")

    return rows


def axis_limits(values):
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return lo - 0.05, hi + 0.05
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def make_pgfplots(rows, args):
    values = [r["value"] for r in rows]
    steps = [r["step_k"] for r in rows]

    ymin, ymax = axis_limits(values)
    xmin, xmax = min(steps), max(steps)

    ylabel = "Validation perplexity" if args.metric == "ppl" else "Validation LM loss"

    lines = []
    lines.append(r"\begin{tikzpicture}")
    lines.append(r"\begin{axis}[")
    lines.append(r"    width=\linewidth,")
    lines.append(f"    height={args.height},")
    lines.append(r"    xlabel={Training steps (k)},")
    lines.append(f"    ylabel={{{ylabel}}},")
    lines.append(f"    xmin={xmin:.0f}, xmax={xmax:.0f},")
    lines.append(f"    ymin={ymin:.6f}, ymax={ymax:.6f},")
    lines.append(r"    grid=both,")
    lines.append(r"    major grid style={line width=.2pt,draw=gray!25},")
    lines.append(r"    minor grid style={line width=.1pt,draw=gray!10},")
    lines.append(r"    legend style={font=\scriptsize, draw=none, fill=none, at={(0.02,0.98)}, anchor=north west},")
    lines.append(r"    tick label style={font=\scriptsize},")
    lines.append(r"    label style={font=\small},")
    lines.append(r"    legend columns=2,")
    lines.append(r"    mark size=1.35pt,")
    lines.append(r"]")

    for run in args.runs:
        rr = sorted([r for r in rows if r["run"] == run], key=lambda x: x["step"])
        if not rr:
            continue

        coords = " ".join(f"({r['step_k']:.0f},{r['value']:.6f})" for r in rr)

        lines.append(
            "\\addplot+"
            f"[color={COLORS.get(run, 'black')}, "
            f"{STYLES.get(run, 'solid')}, "
            f"mark={MARKS.get(run, '*')}] "
            f"coordinates {{{coords}}};"
        )
        lines.append(f"\\addlegendentry{{{PRETTY.get(run, run)}}}")

    lines.append(r"\end{axis}")
    lines.append(r"\end{tikzpicture}")
    lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Missing input CSV: {input_path}")

    if args.out is None:
        suffix = f"seq2048_{args.metric}"
        if args.zoom_from_step:
            suffix += f"_from{args.zoom_from_step}"
        out_path = Path("figures") / f"training_curve_{suffix}_pgfplots.tex"
    else:
        out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path, args.metric, args.runs, args.zoom_from_step)
    out_path.write_text(make_pgfplots(rows, args))

    print(f"input={input_path}")
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()