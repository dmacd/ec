from __future__ import annotations

import math
from typing import Any, Dict, FrozenSet, Optional, Protocol, Sequence, Tuple

from .rbii_state import RBIIState


class RBIIWindowLossModel(Protocol):
    def baseline_bits(
        self,
        *,
        state: RBIIState,
        cfg: Any,
        start_timestep: int,
        end_timestep: int,
    ) -> float:
        ...

    def loss_bits(
        self,
        *,
        prediction: Any,
        observed: str,
        state: RBIIState,
        cfg: Any,
        timestep: int,
    ) -> Optional[float]:
        ...

    def mixture_predict_symbol(
        self,
        *,
        weighted_predictions: Sequence[Tuple[float, Any]],
        state: RBIIState,
        cfg: Any,
        timestep: int,
    ) -> Optional[str]:
        ...


class CategoricalLogLossModel:
    """
    General categorical log-loss accounting for symbol prediction.

    Supported predictor outputs:
    - symbol (str): treated as a deterministic categorical distribution with
      epsilon smoothing over the configured alphabet.
    - dict[str, float]: treated as unnormalized categorical mass over the
      configured alphabet; normalized.
    - sequence[float]: interpreted against cfg.alphabet order when lengths match.
    """

    def baseline_bits(
        self,
        *,
        state: RBIIState,
        cfg: Any,
        start_timestep: int,
        end_timestep: int,
    ) -> float:
        _ = state
        if end_timestep < start_timestep:
            return 0.0
        window_len = int(end_timestep - start_timestep + 1)
        return float(window_len) * float(cfg.baseline_bits_per_symbol)

    def loss_bits(
        self,
        *,
        prediction: Any,
        observed: str,
        state: RBIIState,
        cfg: Any,
        timestep: int,
    ) -> Optional[float]:
        _ = state
        _ = timestep
        if observed not in cfg.alphabet_set:
            raise ValueError(f"Observed symbol {observed!r} not present in configured alphabet.")

        dist = self._distribution(
            prediction=prediction,
            alphabet=cfg.alphabet,
            alphabet_set=cfg.alphabet_set,
            cfg=cfg,
        )
        if dist is None:
            return None

        p = float(dist.get(observed, 0.0))
        p_floor = max(float(getattr(cfg, "min_probability", 1e-12)), 1e-300)
        p = max(p, p_floor)
        return -math.log2(p)

    def mixture_predict_symbol(
        self,
        *,
        weighted_predictions: Sequence[Tuple[float, Any]],
        state: RBIIState,
        cfg: Any,
        timestep: int,
    ) -> Optional[str]:
        _ = state
        _ = timestep
        if not weighted_predictions:
            return None

        mixture: Dict[str, float] = {s: 0.0 for s in cfg.alphabet}
        have_valid = False
        for w, pred in weighted_predictions:
            if w <= 0.0:
                continue
            dist = self._distribution(
                prediction=pred,
                alphabet=cfg.alphabet,
                alphabet_set=cfg.alphabet_set,
                cfg=cfg,
            )
            if dist is None:
                continue
            have_valid = True
            for s, p in dist.items():
                mixture[s] = mixture.get(s, 0.0) + float(w) * float(p)

        if not have_valid:
            return None
        return max(mixture.items(), key=lambda kv: kv[1])[0]

    def _distribution(
        self,
        *,
        prediction: Any,
        alphabet: Tuple[str, ...],
        alphabet_set: FrozenSet[str],
        cfg: Any,
    ) -> Optional[Dict[str, float]]:
        if isinstance(prediction, str):
            if prediction not in alphabet_set:
                return None
            eps = float(getattr(cfg, "deterministic_smoothing_eps", 1e-3))
            eps = min(max(eps, 0.0), 1.0)
            k = len(alphabet)
            if k == 1:
                return {alphabet[0]: 1.0}
            off = eps / float(k - 1)
            out = {s: off for s in alphabet}
            out[prediction] = 1.0 - eps
            return out

        if isinstance(prediction, dict):
            raw: Dict[str, float] = {s: 0.0 for s in alphabet}
            for key, value in prediction.items():
                if key not in alphabet_set:
                    return None
                try:
                    mass = float(value)
                except Exception:
                    return None
                if not math.isfinite(mass) or mass < 0.0:
                    return None
                raw[key] = mass
            total = sum(raw.values())
            if total <= 0.0:
                return None
            return {s: (raw[s] / total) for s in alphabet}

        if isinstance(prediction, (list, tuple)):
            if len(prediction) != len(alphabet):
                return None
            values = []
            for x in prediction:
                try:
                    xv = float(x)
                except Exception:
                    return None
                if not math.isfinite(xv) or xv < 0.0:
                    return None
                values.append(xv)
            total = sum(values)
            if total <= 0.0:
                return None
            return {s: (v / total) for s, v in zip(alphabet, values)}

        return None
