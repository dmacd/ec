from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .types import argmax_label, safe_log2_prob


@dataclass
class ContextStats:
    n: int = 0
    correct: int = 0
    bits_sum: float = 0.0


@dataclass
class OnlineMNISTMetrics:
    contexts: List[str] = field(default_factory=list)
    correct_flags: List[int] = field(default_factory=list)
    bits: List[float] = field(default_factory=list)

    per_context: Dict[str, ContextStats] = field(default_factory=dict)

    def update(self, context: str, y_true: int, dist: torch.Tensor) -> None:
        pred = argmax_label(dist)
        is_correct = int(pred == int(y_true))
        log2p = safe_log2_prob(dist[int(y_true)])
        bits = -log2p

        self.contexts.append(str(context))
        self.correct_flags.append(is_correct)
        self.bits.append(bits)

        stats = self.per_context.setdefault(str(context), ContextStats())
        stats.n += 1
        stats.correct += is_correct
        stats.bits_sum += bits

    def _segments(self) -> List[tuple[str, int, int]]:
        if not self.contexts:
            return []

        out: List[tuple[str, int, int]] = []
        start = 0
        cur = self.contexts[0]
        for i in range(1, len(self.contexts)):
            if self.contexts[i] != cur:
                out.append((cur, start, i))
                start = i
                cur = self.contexts[i]
        out.append((cur, start, len(self.contexts)))
        return out

    def reacquisition_report(self, window: int = 20, tolerance: float = 0.02) -> Dict[str, object]:
        segments = self._segments()
        baseline: Dict[str, Dict[str, float]] = {}
        returns: List[Dict[str, object]] = []

        for context, start, end in segments:
            seg_correct = self.correct_flags[start:end]
            seg_bits = self.bits[start:end]
            seg_n = len(seg_correct)
            if seg_n == 0:
                continue

            seg_acc = sum(seg_correct) / float(seg_n)
            seg_bits_mean = sum(seg_bits) / float(seg_n)

            if context not in baseline:
                baseline[context] = {
                    "acc": seg_acc,
                    "bits_mean": seg_bits_mean,
                    "start": start,
                    "end": end,
                }
                continue

            target = baseline[context]["acc"] - float(tolerance)
            delay: Optional[int] = None
            for j in range(seg_n):
                w_start = max(0, j - window + 1)
                w = seg_correct[w_start : (j + 1)]
                if (sum(w) / float(len(w))) >= target:
                    delay = j
                    break

            expected_bits = baseline[context]["bits_mean"] * seg_n
            excess_bits = sum(seg_bits) - expected_bits

            returns.append(
                {
                    "context": context,
                    "segment_start": start,
                    "segment_end": end,
                    "segment_n": seg_n,
                    "segment_acc": seg_acc,
                    "segment_bits_mean": seg_bits_mean,
                    "baseline_acc": baseline[context]["acc"],
                    "baseline_bits_mean": baseline[context]["bits_mean"],
                    "reacquisition_delay": delay,
                    "excess_bits": excess_bits,
                }
            )

        return {
            "segments": [
                {"context": c, "start": s, "end": e, "n": e - s}
                for c, s, e in segments
            ],
            "returns": returns,
        }

    def summary(self) -> Dict[str, object]:
        total_n = len(self.correct_flags)
        total_correct = sum(self.correct_flags)
        total_bits = sum(self.bits)

        per_context = {}
        for c, stats in self.per_context.items():
            n = max(stats.n, 1)
            per_context[c] = {
                "n": stats.n,
                "accuracy": stats.correct / float(n),
                "mean_logloss_bits": stats.bits_sum / float(n),
            }

        return {
            "n": total_n,
            "accuracy": (total_correct / float(total_n)) if total_n > 0 else 0.0,
            "mean_logloss_bits": (total_bits / float(total_n)) if total_n > 0 else 0.0,
            "per_context": per_context,
            "reacquisition": self.reacquisition_report(),
        }
