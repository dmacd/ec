from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from dreamcoder.utilities import NEGATIVEINFINITY

from .types import coerce_prediction, prediction_to_distribution, safe_log2_prob


@dataclass
class MNISTLogLossLikelihoodModel:
    """
    Log-loss based likelihood model for enumerateForTasks.

    Returns likelihood = -mean_logloss_bits, so higher is better.
    """

    timeout: Optional[float] = 0.05
    label_smoothing_eps: float = 1e-3

    def score(self, program, task):
        started = time.perf_counter()

        try:
            fn = program.evaluate([])
        except Exception:
            return False, NEGATIVEINFINITY

        bits_sum = 0.0
        n = 0

        for xs, y in task.examples:
            if self.timeout is not None and (time.perf_counter() - started) > self.timeout:
                return False, NEGATIVEINFINITY

            try:
                out = fn
                for arg in xs:
                    out = out(arg)

                pred = coerce_prediction(out)
                dist = prediction_to_distribution(pred, eps=self.label_smoothing_eps)
                bits = -safe_log2_prob(dist[int(y)])
            except Exception:
                return False, NEGATIVEINFINITY

            if not math.isfinite(bits):
                return False, NEGATIVEINFINITY

            bits_sum += bits
            n += 1

        if n == 0:
            return False, NEGATIVEINFINITY

        mean_bits = bits_sum / float(n)
        if not math.isfinite(mean_bits):
            return False, NEGATIVEINFINITY

        return True, -mean_bits
