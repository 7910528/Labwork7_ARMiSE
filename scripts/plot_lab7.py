"""Рисунок 1: task-level частка пройдених тестів за форматами (парні лінії = та сама задача)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

CONDITIONS = ["paragraph", "list", "sections"]
LABELS = {"paragraph": "Абзац", "list": "Список", "sections": "Секції"}
INK, MUTED, GRID, DOT = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"

res = Path(sys.argv[1] if len(sys.argv) > 1 else "../results")
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True, dpi=200)
for ax, key, title in [(axes[0], "strict_all", "Основне правило вилучення коду (strict)"),
                       (axes[1], "lenient_all", "Альтернативне правило (lenient)")]:
    wide = pd.read_csv(res / f"task_level_{key}_q.csv", index_col=0)[CONDITIONS]
    tasks = list(wide.index)
    for i, t in enumerate(tasks):
        off = (i - (len(tasks) - 1) / 2) * 0.025
        xs = [k + off for k in range(3)]
        ax.plot(xs, wide.loc[t].values, color=GRID, lw=1.2, zorder=1)
        ax.scatter(xs, wide.loc[t].values, s=26, color=DOT, edgecolor="white", linewidth=1, zorder=2)
    means = wide.mean()
    for k, c in enumerate(CONDITIONS):
        ax.hlines(means[c], k - 0.22, k + 0.22, color=INK, lw=2.2, zorder=3)
        ax.text(k + 0.25, means[c], f"{means[c]:.2f}".replace(".", ","), va="center", fontsize=9, color=INK)
    ax.set_xticks(range(3), [LABELS[c] for c in CONDITIONS])
    ax.set_xlim(-0.45, 2.6)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
axes[0].set_ylabel("Частка пройдених тестів (середнє 3 повторів)", fontsize=9, color=MUTED)
fig.text(0.5, 0.005, "Точка — задача MBPP (n = 10); сіра лінія з'єднує ту саму задачу; чорна риска — середнє по задачах",
         ha="center", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(res / "figure1_task_level_q.png")
print("saved", res / "figure1_task_level_q.png")
