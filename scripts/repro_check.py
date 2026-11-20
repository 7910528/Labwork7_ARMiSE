"""Перевірка відтворюваності: повторно генерує вибрані комбінації з тими самими seed і порівнює з сирими відповідями."""
import json
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DTYPE, GEN_PARAMS, MODEL_NAME, MODEL_REVISION, NUM_THREADS

raw_path, k = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3
recs = [json.loads(line) for line in open(raw_path, encoding="utf-8")]
picks = random.Random(7).sample(recs, k)
torch.set_num_threads(NUM_THREADS)
tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, dtype=getattr(torch, DTYPE)).eval()
same = 0
for r in picks:
    enc = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    torch.manual_seed(r["seed"])
    with torch.no_grad():
        g = model.generate(**enc, **GEN_PARAMS, pad_token_id=tok.eos_token_id)
    resp = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    ok = resp == r["response"]
    same += ok
    print(f"task {r['task_id']} {r['condition']} r{r['repeat']} seed {r['seed']}: {'identical' if ok else 'DIFFERENT'}")
print(f"{same}/{k} identical")
