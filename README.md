# ЛР7 — пакет даних і скриптів

Пілотний експеримент: вплив формату prompt (абзац / список / секції) на тестову коректність
Python-коду, згенерованого Qwen2.5-Coder-1.5B-Instruct, на задачах MBPP task_id 11–20
(3 формати × 3 повтори = 90 генерацій). Основний запуск: 14.11.2026, 08:54–09:22 UTC.

## Повторення

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers pandas scipy matplotlib
cd scripts
python3 evaluate.py --reference --tasks 11 12 13 14 15 16 17 18 19 20   # перевірка harness: 3/3 для кожної задачі
python3 generate.py --tasks 11 12 13 14 15 16 17 18 19 20 --repeats 1 2 3 --version v2 \
        --order-seed 20261114 --out ../data/main_raw.jsonl            # ~30 хв на 2 ядрах CPU
python3 evaluate.py --raw ../data/main_raw.jsonl --csv ../data/main_results.csv --log ../data/main_testlog.jsonl
python3 score_lab7.py --self-test
python3 score_lab7.py --csv ../data/main_results.csv --out ../results
python3 plot_lab7.py ../results
python3 repro_check.py ../data/main_raw.jsonl 3                         # повторна генерація з тими самими seed
```

`evaluate.py` використовує `unshare -n` (Linux) для запуску тестів без мережі.

## Уточнення щодо запуску й безпеки

Ці команди розраховані на Linux, оскільки `evaluate.py` використовує модуль `resource` та `unshare`. Їх не можна виконувати безпосередньо у Windows PowerShell; для відтворення потрібне сумісне Linux-середовище.

Оцінювач запускає код у дочірньому процесі з вимкненою мережею, тимчасовою робочою текою та лімітами ресурсів. Це не повна ізоляція контейнером або віртуальною машиною і не підтверджує ізоляцію файлової системи. Не використовуйте скрипт для запуску довільного стороннього коду.

Повторна генерація з тим самим seed була перевірена для трьох випадково вибраних записів; усі три відповіді збіглися до символу. Це підтверджує ці три перевірені випадки, але не гарантує побітову повторюваність усіх генерацій на будь-якому обладнанні.

`scripts/lab6_reference/score_lab6.py` та `lab6-pilot-fixture.csv` включено як інструментарій попередньої лабораторної; для аналізу даних ЛР7 використовується `scripts/score_lab7.py`.

## Файли

- `data/mbpp.jsonl` — копія MBPP (SHA-256 у `data/preregistration.json`)
- `data/preregistration.json` — протокол, зафіксований до запусків
- `data/protocol_amendments.json` — зміни A1–A2 після технічної проби
- `data/trial_v1_*`, `data/trial_v2_*` — технічна проба на task_id 21–23
- `data/main_raw.jsonl` — prompt, сира відповідь і параметри кожної генерації
- `data/main_results.csv` — основний набір даних (словник — у звіті, розд. 8)
- `data/main_testlog.jsonl` — результат кожного тесту
- `data/lab7-data-template.csv` — CSV-шаблон
- `data/repro_check.log` — перевірка відтворюваності
- `results/` — analysis.json, task-level таблиці, рисунок 1
- `scripts/` — common.py, generate.py, evaluate.py, score_lab7.py, plot_lab7.py, repro_check.py
