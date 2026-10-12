"""Compact report figures for the offline 0929-G1 state/expression audit."""
import json
from collections import Counter
from pathlib import Path


def render(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    rows = [json.loads(line) for line in (output / "aligned_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    verdicts = {r["id"]: r for r in (json.loads(line) for line in (output / "judge_results.jsonl").read_text(encoding="utf-8").splitlines())}
    names = ["Consistent", "Partial", "Not observable", "Contradicted"]
    keys = ["consistent", "partially_consistent", "not_observable", "contradicted"]
    colors = ["#2a9d8f", "#e9c46a", "#8d99ae", "#d1495b"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    counts = Counter(v["label"] for v in verdicts.values())
    axes[0].bar(names, [counts[k] for k in keys], color=colors)
    axes[0].set_ylabel("Message × construct judgments")
    axes[0].tick_params(axis="x", labelrotation=20)
    for i, key in enumerate(keys):
        axes[0].text(i, counts[key] + 5, str(counts[key]), ha="center", fontsize=9)
    sections = [("Early", 1, 13), ("Middle", 14, 26), ("Late", 27, 100)]
    obs_rates, scores = [], []
    for _, lower, upper in sections:
        items = [verdicts[r["id"]] for r in rows if lower <= r["session_order"] <= upper and r["id"] in verdicts]
        c = Counter(v["label"] for v in items)
        obs = c["consistent"] + c["partially_consistent"] + c["contradicted"]
        obs_rates.append(obs / len(items) if items else 0)
        scores.append((c["consistent"] + .5 * c["partially_consistent"]) / obs if obs else 0)
    x = list(range(3))
    axes[1].plot(x, obs_rates, "o-", color="#457b9d", label="Observable rate")
    axes[1].plot(x, scores, "s-", color="#2a9d8f", label="Conformity | observable")
    axes[1].set_xticks(x, [name for name, _, _ in sections])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Rate")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].grid(axis="y", alpha=.2)
    destination = output / "conformity_overview.png"
    fig.savefig(destination, dpi=170)
    plt.close(fig)
    return destination


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(render(parser.parse_args().output))
