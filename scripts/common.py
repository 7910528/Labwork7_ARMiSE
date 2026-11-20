"""Спільні визначення для ЛР7: задачі MBPP, шаблони prompt, параметри моделі."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MBPP_PATH = DATA / "mbpp.jsonl"

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
DTYPE = "bfloat16"
# Параметри збігаються з generation_config.json постачальника для цієї ревізії
GEN_PARAMS = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                  repetition_penalty=1.1, max_new_tokens=512)
NUM_THREADS = 2

CONDITIONS = ["paragraph", "list", "sections"]
MAIN_TASKS = list(range(11, 21))      # основна пілотна вибірка: task_id 11–20
TRIAL_TASKS = [21, 22, 23]            # технічна проба поза вибіркою
REPEATS = [1, 2, 3]

# Зафіксовано ДО основного запуску (див. data/preregistration.json):
# задачі, де очікуваний формат результату не випливає з тексту задачі
AMBIGUOUS_TASKS = [13, 15, 16]


def load_mbpp():
    with open(MBPP_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return {r["task_id"]: r for r in rows}


def entry_point(task):
    """Ім'я функції, яку викликають тести: визначена в еталонному коді й ужита в test_list[0]."""
    defined = re.findall(r"^def\s+(\w+)\s*\(", task["code"], flags=re.M)
    called = set(re.findall(r"(\w+)\s*\(", task["test_list"][0]))
    for name in defined:
        if name in called:
            return name
    raise ValueError(f"entry point not found for task {task['task_id']}")


# v1 — шаблони з Додатка А ЛР6 (без імені функції)
TEMPLATES_V1 = {
    "paragraph": 'Implement the following task in Python: "{task}". Return only the function code.',
    "list": 'Implement the following task in Python:\n• "{task}".\n• Return only the function code.',
    "sections": 'Implement the following task in Python:\nTask: "{task}".\nOutput: Return only the function code.',
}

# v2 — після технічної проби: до ВСІХ трьох умов додано однакове речення з іменем функції
TEMPLATES_V2 = {
    "paragraph": ('Implement the following task in Python: "{task}". '
                  'The function must be named {name}. Return only the function code.'),
    "list": ('Implement the following task in Python:\n'
             '• "{task}".\n'
             '• The function must be named {name}.\n'
             '• Return only the function code.'),
    "sections": ('Implement the following task in Python:\n'
                 'Task: "{task}".\n'
                 'Function: The function must be named {name}.\n'
                 'Output: Return only the function code.'),
}


def build_prompt(task, condition, version="v2"):
    tpl = (TEMPLATES_V2 if version == "v2" else TEMPLATES_V1)[condition]
    return tpl.format(task=task["text"].strip(), name=entry_point(task))


def seed_for(task_id, condition, repeat, version="v2"):
    base = 0 if version == "v2" else 500000
    return base + task_id * 100 + CONDITIONS.index(condition) * 10 + repeat
