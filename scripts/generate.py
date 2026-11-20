"""Генерація відповідей моделі за списком завдань (JSONL на виході, з відновленням після збою).

Використання:
  python3 generate.py --tasks 21 22 23 --repeats 1 --version v1 --out ../data/trial_v1_raw.jsonl
  python3 generate.py --tasks 11 ... 20 --repeats 1 2 3 --version v2 --out ../data/main_raw.jsonl --order-seed 20261114
"""
import argparse
import datetime as dt
import json
import platform
import random
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (CONDITIONS, DTYPE, GEN_PARAMS, MODEL_NAME, MODEL_REVISION,
                    NUM_THREADS, build_prompt, load_mbpp, seed_for)


def make_jobs(task_ids, repeats, order_seed):
    """Порядок умов рандомізується в межах кожного блоку «задача × повтор»."""
    rng = random.Random(order_seed)
    jobs = []
    for t in task_ids:
        for r in repeats:
            conds = CONDITIONS[:]
            rng.shuffle(conds)
            for c in conds:
                jobs.append((t, c, r))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, nargs="+", required=True)
    ap.add_argument("--repeats", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--order-seed", type=int, default=20261114)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(NUM_THREADS)
    mbpp = load_mbpp()
    jobs = make_jobs(args.tasks, args.repeats, args.order_seed)

    done = set()
    try:
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                done.add((d["task_id"], d["condition"], d["repeat"]))
    except FileNotFoundError:
        pass

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=MODEL_REVISION,
                                                 dtype=getattr(torch, DTYPE))
    model.eval()

    env = dict(python=platform.python_version(), torch=torch.__version__,
               transformers=transformers.__version__, platform=platform.platform(),
               threads=NUM_THREADS)

    with open(args.out, "a", encoding="utf-8") as out:
        for i, (t, c, r) in enumerate(jobs, 1):
            if (t, c, r) in done:
                continue
            prompt = build_prompt(mbpp[t], c, args.version)
            msgs = [{"role": "user", "content": prompt}]   # без власної системної інструкції
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True)
            seed = seed_for(t, c, r, args.version)
            torch.manual_seed(seed)
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(**enc, **GEN_PARAMS, pad_token_id=tok.eos_token_id)
            new = gen[0][enc["input_ids"].shape[1]:]
            response = tok.decode(new, skip_special_tokens=True)
            rec = dict(task_id=t, condition=c, repeat=r, order_index=i, seed=seed,
                       template_version=args.version, prompt=prompt, response=response,
                       prompt_tokens=int(enc["input_ids"].shape[1]),
                       response_tokens=int(new.shape[0]),
                       gen_seconds=round(time.time() - t0, 2),
                       timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                       model=MODEL_NAME, revision=MODEL_REVISION, dtype=DTYPE,
                       gen_params=GEN_PARAMS, env=env)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{i}/{len(jobs)}] task {t} {c} r{r}: {rec['response_tokens']} tok, "
                  f"{rec['gen_seconds']} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
