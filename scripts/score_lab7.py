"""Аналіз результатів ЛР7 (розвиток score_lab6.py).

  python3 score_lab7.py --csv ../data/main_results.csv --out ../results
  python3 score_lab7.py --self-test

Одиниця аналізу — задача MBPP: повтори спершу усереднюються в межах «задача × умова»,
далі — рівновагове середнє по задачах. 90 генерацій НЕ трактуються як незалежні спостереження.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CONDITIONS = ["paragraph", "list", "sections"]
PAIRS = [("list", "paragraph"), ("sections", "paragraph"), ("sections", "list")]
AMBIGUOUS = [13, 15, 16]


def validate(df, n_repeats=3, expected_tasks=None):
    """Повнота парного факторного плану та коректність значень."""
    errors = []
    need = {"task_id", "condition", "repeat", "passed_tests", "total_tests", "syntax_ok"}
    if missing := need - set(df.columns):
        return [f"missing columns: {sorted(missing)}"]
    if df[list(need)].isna().any().any():
        errors.append("required fields contain missing values")
    if not set(df["condition"]).issubset(CONDITIONS):
        errors.append("unknown condition values")
    if df.duplicated(["task_id", "condition", "repeat"]).any():
        errors.append("duplicated task×condition×repeat")
    if expected_tasks is not None:
        found = set(df["task_id"].astype(int))
        expected = set(expected_tasks)
        if found != expected:
            errors.append(
                f"task IDs differ from protocol: missing={sorted(expected - found)}, "
                f"unexpected={sorted(found - expected)}"
            )
    for t, g in df.groupby("task_id"):
        if len(g) != len(CONDITIONS) * n_repeats:
            errors.append(f"task {t}: {len(g)} rows instead of {len(CONDITIONS) * n_repeats}")
        if set(g["condition"]) != set(CONDITIONS):
            errors.append(f"task {t}: missing or unexpected conditions")
        for condition, cell in g.groupby("condition"):
            if set(cell["repeat"].astype(int)) != set(range(1, n_repeats + 1)):
                errors.append(f"task {t}, condition {condition}: repeats differ from protocol")
        if g["total_tests"].nunique() != 1:
            errors.append(f"task {t}: total_tests not constant")
    if ((df["passed_tests"] < 0) | (df["passed_tests"] > df["total_tests"])).any():
        errors.append("passed_tests out of range")
    if (df["total_tests"] <= 0).any():
        errors.append("total_tests must be positive")
    syntax_values = set(df["syntax_ok"].astype(str).str.lower())
    if not syntax_values.issubset({"true", "false", "1", "0"}):
        errors.append("syntax_ok must contain only true/false values")
    bad_syntax = df[(df["syntax_ok"].astype(str).str.lower() == "false") & (df["passed_tests"] > 0)]
    if len(bad_syntax):
        errors.append("passed_tests > 0 for syntax_ok=false")
    return errors


def task_level(df, passed_col="passed_tests"):
    d = df.copy()
    d["q"] = d[passed_col] / d["total_tests"]
    d["f"] = (d[passed_col] == d["total_tests"]).astype(float)
    return d.groupby(["task_id", "condition"])[["q", "f"]].mean().reset_index()


def condition_summary(tl):
    s = tl.groupby("condition")[["q", "f"]].agg(["mean", "std"])
    return s.reindex(CONDITIONS)


def paired(tl, metric):
    wide = tl.pivot(index="task_id", columns="condition", values=metric)[CONDITIONS]
    out = []
    for a, b in PAIRS:
        diff = wide[a] - wide[b]
        out.append(dict(pair=f"{a}−{b}", metric=metric, mean_diff=diff.mean(), median_diff=diff.median(),
                        better=int((diff > 1e-9).sum()), equal=int((diff.abs() <= 1e-9).sum()),
                        worse=int((diff < -1e-9).sum()), n=len(diff)))
    return pd.DataFrame(out), wide


def friedman(wide):
    if (wide.nunique(axis=1) == 1).all():
        return dict(stat=float("nan"), p=float("nan"), note="усі значення однакові")
    r = stats.friedmanchisquare(*[wide[c] for c in CONDITIONS])
    return dict(stat=float(r.statistic), p=float(r.pvalue), note="пошуковий, не підтверджувальний")


def loo(tl, metric):
    """Leave-one-task-out: діапазон середньої парної різниці при вилученні кожної задачі."""
    res = []
    tasks = sorted(tl["task_id"].unique())
    for a, b in PAIRS:
        vals = []
        for t in tasks:
            sub = tl[tl["task_id"] != t]
            w = sub.pivot(index="task_id", columns="condition", values=metric)
            vals.append((w[a] - w[b]).mean())
        full = tl.pivot(index="task_id", columns="condition", values=metric)
        fm = (full[a] - full[b]).mean()
        res.append(dict(pair=f"{a}−{b}", metric=metric, full=fm, loo_min=min(vals), loo_max=max(vals),
                        sign_flips=int(sum((v > 1e-9) != (fm > 1e-9) or (v < -1e-9) != (fm < -1e-9) for v in vals))))
    return pd.DataFrame(res)


def per_repeat(df, passed_col="passed_tests"):
    rows = []
    for r, g in df.groupby("repeat"):
        tl = task_level(g, passed_col)
        s = tl.groupby("condition")[["q", "f"]].mean().reindex(CONDITIONS)
        for c in CONDITIONS:
            rows.append(dict(repeat=r, condition=c, q=s.loc[c, "q"], f=s.loc[c, "f"]))
    return pd.DataFrame(rows)


def variability(df):
    """Мінливість між повторами (всередині задача×умова) проти мінливості між умовами (всередині задачі)."""
    d = df.copy()
    d["q"] = d["passed_tests"] / d["total_tests"]
    within = d.groupby(["task_id", "condition"])["q"].std(ddof=0).groupby("task_id").mean()
    between = d.groupby(["task_id", "condition"])["q"].mean().groupby("task_id").std(ddof=0)
    unstable = int((d.groupby(["task_id", "condition"])["q"].nunique() > 1).sum())
    return dict(mean_within_sd=float(within.mean()), mean_between_sd=float(between.mean()),
                cells_with_varying_repeats=unstable, cells_total=int(d.groupby(["task_id", "condition"]).ngroups),
                per_task=pd.DataFrame({"within_sd": within, "between_sd": between}).reset_index())


def analyze(df, out):
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    errs = validate(df, expected_tasks=range(11, 21))
    report["validation_errors"] = errs
    if errs:
        print("VALIDATION FAILED:", errs)
        return report

    variants = {"strict": "passed_tests", "lenient": "lenient_passed_tests"}
    for name, col in variants.items():
        if col not in df.columns:
            continue
        for subset_name, sub in [("all", df), ("no_ambiguous", df[~df["task_id"].isin(AMBIGUOUS)])]:
            tl = task_level(sub, col)
            key = f"{name}_{subset_name}"
            cs = condition_summary(tl)
            report[key] = {"n_tasks": int(tl["task_id"].nunique()),
                           "conditions": {c: {"q_mean": cs.loc[c, ("q", "mean")], "q_sd": cs.loc[c, ("q", "std")],
                                              "f_mean": cs.loc[c, ("f", "mean")], "f_sd": cs.loc[c, ("f", "std")]}
                                          for c in CONDITIONS}}
            for metric in ["q", "f"]:
                pdiff, wide = paired(tl, metric)
                report[key][f"paired_{metric}"] = pdiff.to_dict("records")
                report[key][f"friedman_{metric}"] = friedman(wide)
                wide.round(4).to_csv(out / f"task_level_{key}_{metric}.csv")
            if subset_name == "all":
                report[key]["loo"] = pd.concat([loo(tl, "q"), loo(tl, "f")]).to_dict("records")
                tl.to_csv(out / f"task_level_{key}.csv", index=False)
        report[f"{name}_per_repeat"] = per_repeat(df, col).to_dict("records")
        # мікроагрегування (усі генерації разом) як альтернатива рівноваговому
        d = df.assign(q=df[col] / df["total_tests"], f=(df[col] == df["total_tests"]).astype(float))
        report[f"{name}_pooled"] = d.groupby("condition")[["q", "f"]].mean().reindex(CONDITIONS).to_dict("index")

    v = variability(df)
    v["per_task"].round(4).to_csv(out / "variability_per_task.csv", index=False)
    report["variability"] = {k: val for k, val in v.items() if k != "per_task"}
    if "failure_type" in df.columns:
        report["failure_types"] = pd.crosstab(df["failure_type"], df["condition"]).reindex(columns=CONDITIONS, fill_value=0).to_dict()
    for col in ["prompt_chars", "prompt_words", "prompt_tokens", "response_tokens"]:
        if col in df.columns:
            report[f"mean_{col}"] = df.groupby("condition")[col].mean().reindex(CONDITIONS).round(1).to_dict()
    with open(out / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    return report


def self_test():
    """Синтетичні дані з відомими агрегатами; не є задачами MBPP і не є генераціями LLM."""
    rows = []
    passed = {1: {"paragraph": [3, 3, 0], "list": [3, 3, 3], "sections": [0, 0, 0]},
              2: {"paragraph": [0, 0, 0], "list": [2, 2, 2], "sections": [4, 4, 4]}}
    total = {1: 3, 2: 4}
    for t in passed:
        for c in CONDITIONS:
            for r, p in enumerate(passed[t][c], 1):
                rows.append(dict(task_id=t, condition=c, repeat=r, passed_tests=p, total_tests=total[t],
                                 syntax_ok="true" if p > 0 or c != "sections" else "false",
                                 lenient_passed_tests=p))
    df = pd.DataFrame(rows)
    assert validate(df) == [], validate(df)
    tl = task_level(df)
    s = tl.groupby("condition")[["q", "f"]].mean()
    assert abs(s.loc["paragraph", "q"] - (2 / 3) / 2) < 1e-9          # (0.667 + 0) / 2
    assert abs(s.loc["list", "q"] - (1 + 0.5) / 2) < 1e-9             # рівновага задач, а не 15/21
    assert abs(s.loc["sections", "f"] - 0.5) < 1e-9
    pdiff, _ = paired(tl, "q")
    assert pdiff.loc[0, "better"] == 2                                 # list > paragraph в обох задачах
    broken = df.drop(index=0)
    assert any("rows instead" in e for e in validate(broken))          # неповний план виявлено
    dup = pd.concat([df, df.iloc[[0]]])
    assert any("duplicated" in e for e in validate(dup))
    print("self-test OK: агрегування, парні різниці та валідація неповного/дубльованого плану працюють")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--out", default="../results")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        sys.exit(0)
    rep = analyze(pd.read_csv(a.csv), Path(a.out))
    print(json.dumps({k: rep[k] for k in rep if k.startswith("strict_all")}, ensure_ascii=False, indent=1, default=float)[:3000])
