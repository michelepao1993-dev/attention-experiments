#!/usr/bin/env python3
import argparse
import csv
import glob
import math
import os
import re
from pathlib import Path


def latest(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def extract_header(text, key):
    m = re.search(r"^" + re.escape(key) + r"=(.*)$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else ""


def parse_validation_line(text):
    lines = [ln for ln in text.splitlines() if "validation loss at iteration" in ln]
    if not lines:
        return {}

    line = lines[-1]
    out = {"validation_line": line}

    # Matches:
    # lm loss value: 2.615765E+00
    # mosar/cost_loss value: 2.187655E-01
    value_re = re.compile(r"([A-Za-z0-9_/ ]+?) value: ([0-9.Ee+-]+)")
    ppl_re = re.compile(r"([A-Za-z0-9_/ ]+?) PPL: ([0-9.Ee+-]+)")

    for key, val in value_re.findall(line):
        norm = key.strip().replace(" ", "_").replace("/", "_")
        out[norm] = float(val)

    for key, val in ppl_re.findall(line):
        norm = key.strip().replace(" ", "_").replace("/", "_") + "_ppl"
        out[norm] = float(val)

    return out


def parse_log(path):
    if path is None:
        return None

    text = Path(path).read_text(errors="replace")
    vals = parse_validation_line(text)

    passed = (
        "MOSAR MAIN EVAL PASSED" in text
        or "MOSAR POSTTRAINING EVAL PASSED" in text
    )

    vals.update({
        "log": path,
        "passed": passed,
        "run_id_header": extract_header(text, "run_id"),
        "trajectory": extract_header(text, "trajectory"),
        "sequence_length": extract_header(text, "sequence_length"),
        "routing_inference": extract_header(text, "routing_inference"),
        "eval_iters": extract_header(text, "eval_iters"),
        "out": extract_header(text, "out"),
    })
    return vals


def fmt(x, nd=6):
    if x is None or x == "":
        return ""
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="main_v1_seed6_500m_20260721")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--runs", nargs="+", default=["mosar0"])
    ap.add_argument("--main-scratch", default=os.environ.get("RUN_ROOT", os.path.join(os.environ.get("SCRATCH_ROOT", "."), "mosar_main_v1")))
    ap.add_argument("--post-scratch", default=os.environ.get("POST_RUN_ROOT", os.path.join(os.environ.get("SCRATCH_ROOT", "."), "mosar_posttraining_v1")))
    args = ap.parse_args()

    rows = []

    for run in args.runs:
        sources = [
            (
                "main_soft",
                latest(f"{args.main_scratch}/logs/{args.group}_eval_step_{args.step}_{run}_*.out"),
            ),
            (
                "post_soft",
                latest(f"{args.post_scratch}/logs/{args.group}_posteval_step_{args.step}_seq_{args.seq}_soft_{run}_*.out"),
            ),
            (
                "post_hard_top1",
                latest(f"{args.post_scratch}/logs/{args.group}_posteval_step_{args.step}_seq_{args.seq}_hard_top1_{run}_*.out"),
            ),
        ]

        for source, log in sources:
            parsed = parse_log(log)
            if parsed is None:
                rows.append({
                    "run": run,
                    "source": source,
                    "found": False,
                    "log": "",
                })
                continue

            def metric(*names):
                for name in names:
                    value = parsed.get(name, "")
                    if value != "":
                        return value
                return ""

            row = {
                "run": run,
                "source": source,
                "found": True,
                "passed": parsed.get("passed", False),
                "lm_loss": parsed.get("lm_loss", ""),
                "ppl": parsed.get("lm_loss_ppl", ""),
                "cost": metric("mosar_cost_loss"),
                "q_cost": metric("mosar_query_cost"),
                "k_cost": metric("mosar_key_cost"),
                "q_entropy": metric("mosar_query_entropy"),
                "k_entropy": metric("mosar_key_entropy"),
                "q_S": metric("mosar_query_regime_0"),
                "q_M": metric("mosar_query_regime_1"),
                "q_G": metric("mosar_query_regime_2"),
                "k_S": metric("mosar_key_regime_0"),
                "k_M": metric("mosar_key_regime_1"),
                "k_G": metric("mosar_key_regime_2"),
                "sequence_length": parsed.get("sequence_length", ""),
                "routing_inference": parsed.get("routing_inference", ""),
                "eval_iters": parsed.get("eval_iters", ""),
                "log": parsed.get("log", ""),
            }
            rows.append(row)

    # Compute deltas hard vs main_soft and hard vs post_soft when available.
    by_run_source = {(r["run"], r["source"]): r for r in rows if r.get("found")}
    for r in rows:
        if not r.get("found"):
            r["delta_ppl_vs_main_soft"] = ""
            r["delta_pct_vs_main_soft"] = ""
            r["delta_ppl_vs_post_soft"] = ""
            r["delta_pct_vs_post_soft"] = ""
            continue

        ppl = r.get("ppl")
        if not isinstance(ppl, float):
            r["delta_ppl_vs_main_soft"] = ""
            r["delta_pct_vs_main_soft"] = ""
            r["delta_ppl_vs_post_soft"] = ""
            r["delta_pct_vs_post_soft"] = ""
            continue

        main_ref = by_run_source.get((r["run"], "main_soft"), {})
        post_ref = by_run_source.get((r["run"], "post_soft"), {})

        for label, ref in [("main_soft", main_ref), ("post_soft", post_ref)]:
            ref_ppl = ref.get("ppl")
            if isinstance(ref_ppl, float):
                d = ppl - ref_ppl
                pct = 100.0 * d / ref_ppl
                r[f"delta_ppl_vs_{label}"] = d
                r[f"delta_pct_vs_{label}"] = pct
            else:
                r[f"delta_ppl_vs_{label}"] = ""
                r[f"delta_pct_vs_{label}"] = ""

    out_dir = Path(args.post_scratch) / "evals" / args.group / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"posttraining_eval_summary_step_{args.step}_seq_{args.seq}.csv"

    fieldnames = [
        "run", "source", "found", "passed",
        "lm_loss", "ppl",
        "delta_ppl_vs_main_soft", "delta_pct_vs_main_soft",
        "delta_ppl_vs_post_soft", "delta_pct_vs_post_soft",
        "cost", "q_cost", "k_cost",
        "q_entropy", "k_entropy",
        "q_S", "q_M", "q_G",
        "k_S", "k_M", "k_G",
        "sequence_length", "routing_inference", "eval_iters",
        "log",
    ]

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"saved_csv={out_csv}")
    print()
    print(",".join(fieldnames))
    for r in rows:
        print(",".join(fmt(r.get(k), 6) for k in fieldnames))


if __name__ == "__main__":
    main()
