#!/usr/bin/env python3
"""Summarize test outcomes from the paired MBPP prompt-format experiment.

This tool processes recorded test counts; it does not execute generated code.
Run generated code only in the isolated environment specified in the protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


CONDITIONS = ("paragraph", "list", "sections")
REPEATS = (1, 2, 3)
PILOT_TASK_IDS = tuple(str(task_id) for task_id in range(11, 21))
FIELDS = (
    "task_id",
    "condition",
    "repeat",
    "passed_tests",
    "total_tests",
    "syntax_ok",
)
FIXTURE = Path(__file__).with_name("lab6-pilot-fixture.csv")


class DataError(ValueError):
    """Raised when the measurement sheet does not match the protocol."""


def parse_bool(value: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise DataError(f"Row {row_number}: syntax_ok must be true or false.")


def read_rows(path: Path) -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise DataError(f"Missing CSV column(s): {', '.join(missing)}.")

        for row_number, raw in enumerate(reader, start=2):
            task_id = (raw.get("task_id") or "").strip()
            condition = (raw.get("condition") or "").strip().lower()
            if not task_id:
                raise DataError(f"Row {row_number}: task_id cannot be blank.")
            if condition not in CONDITIONS:
                raise DataError(
                    f"Row {row_number}: condition must be one of "
                    f"{', '.join(CONDITIONS)}."
                )
            try:
                repeat = int(raw["repeat"])
                passed = int(raw["passed_tests"])
                total = int(raw["total_tests"])
            except (TypeError, ValueError) as error:
                raise DataError(
                    f"Row {row_number}: repeat and test counts must be integers."
                ) from error
            if repeat not in REPEATS:
                raise DataError(f"Row {row_number}: repeat must be 1, 2, or 3.")
            if total <= 0 or passed < 0 or passed > total:
                raise DataError(
                    f"Row {row_number}: require total_tests > 0 and "
                    "0 <= passed_tests <= total_tests."
                )
            syntax_ok = parse_bool(raw["syntax_ok"], row_number)
            if not syntax_ok and passed != 0:
                raise DataError(
                    f"Row {row_number}: syntax-invalid code must have "
                    "passed_tests=0."
                )
            observations.append(
                {
                    "task_id": task_id,
                    "condition": condition,
                    "repeat": repeat,
                    "passed_tests": passed,
                    "total_tests": total,
                    "syntax_ok": syntax_ok,
                }
            )

    if not observations:
        raise DataError("The CSV contains no observations.")
    return observations


def summarize(
    observations: list[dict[str, object]], allow_small: bool = False
) -> dict[str, object]:
    tasks = sorted({str(item["task_id"]) for item in observations})
    if not allow_small and tuple(tasks) != PILOT_TASK_IDS:
        raise DataError(
            "The pilot must contain exactly MBPP test task_ids 11-20; "
            f"found: {', '.join(tasks)}."
        )

    cells: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    seen: set[tuple[str, str, int]] = set()
    for item in observations:
        task_id = str(item["task_id"])
        condition = str(item["condition"])
        repeat = int(item["repeat"])
        key = (task_id, condition, repeat)
        if key in seen:
            raise DataError(
                f"Duplicate observation for task={task_id}, "
                f"condition={condition}, repeat={repeat}."
            )
        seen.add(key)
        cells[(task_id, condition)].append(item)

    expected = {
        (task, condition, repeat)
        for task in tasks
        for condition in CONDITIONS
        for repeat in REPEATS
    }
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        details = []
        if missing:
            details.append(f"missing {len(missing)} required observation(s)")
        if extra:
            details.append(f"found {len(extra)} unexpected observation(s)")
        raise DataError("Incomplete paired design: " + "; ".join(details) + ".")

    task_condition: dict[tuple[str, str], dict[str, float]] = {}
    for task in tasks:
        for condition in CONDITIONS:
            repeats = cells[(task, condition)]
            totals = {int(item["total_tests"]) for item in repeats}
            if len(totals) != 1:
                raise DataError(
                    f"Task {task}: total_tests must be constant across conditions "
                    "and repeats."
                )
            test_fractions = [
                int(item["passed_tests"]) / int(item["total_tests"])
                if bool(item["syntax_ok"])
                else 0.0
                for item in repeats
            ]
            complete = [
                1.0
                if bool(item["syntax_ok"])
                and int(item["passed_tests"]) == int(item["total_tests"])
                else 0.0
                for item in repeats
            ]
            task_condition[(task, condition)] = {
                "mean_test_fraction": sum(test_fractions) / len(REPEATS),
                "complete_pass_fraction": sum(complete) / len(REPEATS),
            }

    condition_results: dict[str, dict[str, object]] = {}
    for condition in CONDITIONS:
        q_values = [
            task_condition[(task, condition)]["mean_test_fraction"]
            for task in tasks
        ]
        full_values = [
            task_condition[(task, condition)]["complete_pass_fraction"]
            for task in tasks
        ]
        condition_results[condition] = {
            "mean_test_fraction": sum(q_values) / len(tasks),
            "complete_pass_fraction": sum(full_values) / len(tasks),
        }

    task_results = [
        {
            "task_id": task,
            "condition": condition,
            **task_condition[(task, condition)],
        }
        for task in tasks
        for condition in CONDITIONS
    ]
    pairwise_differences: dict[str, dict[str, object]] = {}
    for metric in ("mean_test_fraction", "complete_pass_fraction"):
        for left, right in (
            ("list", "paragraph"),
            ("sections", "paragraph"),
            ("sections", "list"),
        ):
            differences = [
                task_condition[(task, left)][metric]
                - task_condition[(task, right)][metric]
                for task in tasks
            ]
            pairwise_differences[f"{left}-{right}:{metric}"] = {
                "mean_difference": sum(differences) / len(differences),
                "task_differences": [
                    {"task_id": task, "difference": difference}
                    for task, difference in zip(tasks, differences)
                ],
            }
    return {
        "n_tasks": len(tasks),
        "n_conditions": len(CONDITIONS),
        "repeats_per_cell": len(REPEATS),
        "n_generations": len(observations),
        "analysis_unit": "MBPP task; repeats are repeated measurements, not independent units",
        "aggregation": (
            "mean the three repeats within each task-condition cell, then take "
            "an equally weighted mean across tasks; compare conditions by "
            "task-level paired differences"
        ),
        "conditions": condition_results,
        "task_results": task_results,
        "pairwise_differences": pairwise_differences,
    }


def self_test() -> None:
    observations = read_rows(FIXTURE)
    result = summarize(observations, allow_small=True)
    expected = {
        "paragraph": (0.5, 1 / 3),
        "list": (19 / 24, 0.5),
        "sections": (19 / 24, 0.5),
    }
    for condition, (mean_q, full_pass) in expected.items():
        actual = result["conditions"][condition]
        if not math.isclose(actual["mean_test_fraction"], mean_q, abs_tol=1e-12):
            raise AssertionError(f"Unexpected mean test fraction for {condition}.")
        if not math.isclose(
            actual["complete_pass_fraction"], full_pass, abs_tol=1e-12
        ):
            raise AssertionError(f"Unexpected full-pass fraction for {condition}.")
    if len(result["pairwise_differences"]) != 6:
        raise AssertionError("Expected paired differences for two metrics and 3 pairs.")
    if any(
        len(values["task_differences"]) != 2
        for values in result["pairwise_differences"].values()
    ):
        raise AssertionError("Each paired comparison must retain task-level values.")

    try:
        summarize(observations)
    except DataError:
        pass
    else:
        raise AssertionError("The tool failed to reject task IDs outside 11-20.")

    incomplete = observations[:-1]
    try:
        summarize(incomplete, allow_small=True)
    except DataError:
        pass
    else:
        raise AssertionError("The tool failed to reject an incomplete paired design.")
    print(
        "PASS: synthetic pilot metrics and incomplete-design validation. "
        "No model output or MBPP data were used."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", nargs="?", type=Path)
    parser.add_argument(
        "--json",
        dest="json_file",
        type=Path,
        help="Write the summary as a JSON file.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run checks against the included synthetic pilot fixture.",
    )
    args = parser.parse_args()

    if args.self_test:
        if args.csv_file:
            parser.error("--self-test does not accept a CSV file.")
        try:
            self_test()
        except (DataError, AssertionError) as error:
            print(f"SELF-TEST FAILED: {error}", file=sys.stderr)
            return 1
        return 0

    if not args.csv_file:
        parser.error("provide a CSV file or use --self-test.")
    try:
        result = summarize(read_rows(args.csv_file))
    except (DataError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print("Descriptive pilot summary (task-weighted, paired design; n=10 tasks):")
    print("condition,mean_test_fraction,complete_pass_fraction")
    for condition in CONDITIONS:
        values = result["conditions"][condition]
        print(
            f"{condition},{values['mean_test_fraction']:.6f},"
            f"{values['complete_pass_fraction']:.6f}"
        )
    print("Pairwise task-level mean differences (left condition minus right):")
    for key, values in result["pairwise_differences"].items():
        print(f"{key}={values['mean_difference']:.6f}")
    print(
        f"tasks={result['n_tasks']}; generations={result['n_generations']}; "
        "repeats per task-condition=3; repeats are not independent units"
    )
    print("This script computes descriptive metrics; it does not test significance.")
    if args.json_file:
        args.json_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"JSON summary written to {args.json_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
