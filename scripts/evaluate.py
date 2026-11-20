"""Оцінювання відповідей моделі тестами MBPP в ізольованому середовищі.

Правило вилучення коду (основне, «strict», з протоколу ЛР6): дозволено лише зняти ОДНУ зовнішню
пару Markdown-огорож ```python … ```; будь-який інший текст лишається й робить код невдалим.
Альтернативне правило («lenient», лише для перевірки стійкості): береться перший блок ``` … ```
у будь-якому місці відповіді.

Кожен тест запускається окремим процесом: без мережі (unshare -n), з `python3 -I`,
порожнім stdin, тимчасовою робочою текою, лімітом пам'яті 1 ГБ, CPU 10 с і тайм-аутом 10 с.

Використання:
  python3 evaluate.py --raw ../data/main_raw.jsonl --csv ../data/main_results.csv --log ../data/main_testlog.jsonl
  python3 evaluate.py --reference --tasks 11 ... 23     # перевірка harness еталонними розв'язками MBPP
"""
import argparse
import ast
import csv
import json
import re
import resource
import subprocess
import sys
import tempfile

from common import entry_point, load_mbpp

TIMEOUT_S = 10
MEM_BYTES = 1024 ** 3
FENCE_RE = re.compile(r"^```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)\n?```$", re.S | re.I)
FIRST_BLOCK_RE = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)```", re.S | re.I)


def extract_strict(response):
    s = response.strip()
    m = FENCE_RE.match(s)
    if m and "```" not in m.group(1):
        return m.group(1), "fenced"
    return s, ("plain" if "```" not in s else "fence_plus_text")


def extract_lenient(response):
    m = FIRST_BLOCK_RE.search(response)
    return (m.group(1), "first_block") if m else (response.strip(), "plain")


def compiles(code):
    try:
        compile(code, "<candidate>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def defines(code, name):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return True
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return True
    return False


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S, TIMEOUT_S))


def run_test(code, setup, test):
    program = f"{code}\n\n{setup}\n\n{test}\n"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            p = subprocess.run(["unshare", "-n", sys.executable, "-I", "-c", program],
                               cwd=tmp, stdin=subprocess.DEVNULL, capture_output=True,
                               text=True, timeout=TIMEOUT_S, preexec_fn=_limits)
        except subprocess.TimeoutExpired:
            return "timeout", ""
    if p.returncode == 0:
        return "pass", ""
    err = p.stderr.strip().splitlines()
    last = err[-1] if err else f"returncode {p.returncode}"
    return ("assert_fail" if last.startswith("AssertionError") else "exception"), last[:200]


def evaluate_code(code, task):
    name = entry_point(task)
    syntax_ok = compiles(code)
    outcomes = []
    if syntax_ok:
        for test in task["test_list"]:
            outcomes.append(run_test(code, task["test_setup_code"], test))
    passed = sum(o[0] == "pass" for o in outcomes)
    total = len(task["test_list"])
    if not syntax_ok:
        ftype = "syntax_error"
    elif not defines(code, name):
        ftype = "missing_function"
    elif passed == total:
        ftype = "none"
    elif any(o[0] == "timeout" for o in outcomes):
        ftype = "timeout"
    elif any(o[0] == "exception" for o in outcomes):
        ftype = "runtime_error"
    else:
        ftype = "wrong_answer"
    return dict(syntax_ok=syntax_ok, passed_tests=passed, total_tests=total,
                full_pass=int(passed == total), failure_type=ftype, outcomes=outcomes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw")
    ap.add_argument("--csv")
    ap.add_argument("--log")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--tasks", type=int, nargs="*")
    args = ap.parse_args()
    mbpp = load_mbpp()

    if args.reference:
        for t in args.tasks:
            r = evaluate_code(mbpp[t]["code"], mbpp[t])
            print(t, entry_point(mbpp[t]), f"{r['passed_tests']}/{r['total_tests']}", r["failure_type"])
        return

    rows, log = [], []
    with open(args.raw, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    for rec in recs:
        task = mbpp[rec["task_id"]]
        code_s, ext_s = extract_strict(rec["response"])
        res_s = evaluate_code(code_s, task)
        code_l, ext_l = extract_lenient(rec["response"])
        res_l = evaluate_code(code_l, task)
        if res_s["failure_type"] == "syntax_error" and res_l["syntax_ok"]:
            res_s["failure_type"] = "non_code_text"
        rows.append(dict(
            task_id=rec["task_id"], condition=rec["condition"], repeat=rec["repeat"],
            passed_tests=res_s["passed_tests"], total_tests=res_s["total_tests"],
            syntax_ok=str(res_s["syntax_ok"]).lower(), full_pass=res_s["full_pass"],
            q=round(res_s["passed_tests"] / res_s["total_tests"], 6),
            failure_type=res_s["failure_type"], extraction=ext_s,
            lenient_passed_tests=res_l["passed_tests"], lenient_syntax_ok=str(res_l["syntax_ok"]).lower(),
            lenient_full_pass=res_l["full_pass"],
            prompt_chars=len(rec["prompt"]), prompt_words=len(rec["prompt"].split()),
            prompt_tokens=rec["prompt_tokens"], response_tokens=rec["response_tokens"],
            order_index=rec["order_index"], seed=rec["seed"],
            template_version=rec["template_version"], timestamp_utc=rec["timestamp_utc"]))
        log.append(dict(task_id=rec["task_id"], condition=rec["condition"], repeat=rec["repeat"],
                        strict=res_s["outcomes"], lenient=res_l["outcomes"]))
    rows.sort(key=lambda d: (d["task_id"], d["condition"], d["repeat"]))
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if args.log:
        with open(args.log, "w", encoding="utf-8") as f:
            for d in log:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"evaluated {len(rows)} generations -> {args.csv}")


if __name__ == "__main__":
    main()
