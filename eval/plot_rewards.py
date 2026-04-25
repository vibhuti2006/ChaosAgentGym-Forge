"""Plot the reward curve from logs/rewards.csv.

Usage:
    python -m eval.plot_rewards --csv logs/rewards.csv --out logs/reward_curve.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np


def _load(csv_path: Path):
    rows = list(csv.DictReader(csv_path.open()))
    episodes = np.arange(len(rows))
    returns = np.array([float(r["return"]) for r in rows])
    success = np.array([int(r["success"]) for r in rows])
    return episodes, returns, success


def _smooth(x: np.ndarray, w: int = 16) -> np.ndarray:
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def plot(csv_path: Path, out_path: Path) -> None:
    episodes, returns, success = _load(csv_path)
    if len(episodes) == 0:
        raise SystemExit(f"no data in {csv_path}")

    smooth_w = max(4, min(32, len(episodes) // 10))
    smooth_ret = _smooth(returns, smooth_w)
    smooth_succ = _smooth(success.astype(float), smooth_w)
    smooth_x = np.arange(smooth_w - 1, len(episodes))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax1.scatter(episodes, returns, s=4, alpha=0.25, color="#888")
    ax1.plot(smooth_x, smooth_ret, color="#1f77b4", linewidth=2,
             label=f"return (smoothed, w={smooth_w})")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("episode return")
    ax1.set_title("ChaosAgentGym — training curve")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    ax2.plot(smooth_x, smooth_succ, color="#2ca02c", linewidth=2)
    ax2.set_ylabel("success rate")
    ax2.set_xlabel("episode")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="logs/rewards.csv")
    p.add_argument("--out", default="logs/reward_curve.png")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    plot(Path(a.csv), Path(a.out))
