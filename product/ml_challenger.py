"""ML only as a Challenger.

Does not replace the evidence architecture. No cosmetic AI Score.
PIT features only; train/validation/OOS split; walk-forward vs Champion.
Rejected unless it shows stable incremental OOS value.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.champion_challenger import SHADOW, ChampionChallengerEngine

ROLE = "CHALLENGER"
CAN_EXECUTE = False


def leakage_check(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    leaks = []
    for i, row in enumerate(rows):
        feat_ts = str(row.get("feature_ts") or row.get("as_of") or "")
        dec_ts = str(row.get("decision_as_of") or row.get("as_of") or "")
        if feat_ts and dec_ts and feat_ts > dec_ts:
            leaks.append(f"row {i} feature_ts {feat_ts} > decision_as_of {dec_ts}")
        if row.get("future_return") not in (None, "") and "future_return" in (row.get("features") or {}):
            leaks.append(f"row {i} future_return leaked into features")
    return leaks


def _dot(weights: Mapping[str, float], features: Mapping[str, Any]) -> float:
    total = 0.0
    for key, w in weights.items():
        try:
            total += float(w) * float(features.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
    return total


class MlChallenger:
    """Linear PIT scorer. Starts SHADOW. Never writes an AI Score onto cards."""

    def __init__(self, challenger_id: str = "ml_ranker_v1", *, path=None) -> None:
        self.challenger_id = challenger_id
        self.engine = ChampionChallengerEngine(path)
        self.weights: dict[str, float] = {}
        self.can_execute = CAN_EXECUTE
        self.role = ROLE

    def register(self, *, hypothesis: str, features: Sequence[str]) -> dict[str, Any]:
        return self.engine.register(
            challenger_id=self.challenger_id,
            hypothesis=hypothesis,
            changed_behavior="ml_finalist_ranking",
            rules={"kind": "linear_ml", "features": list(features), "no_ai_score": True},
            training_data="train",
            validation_data="val",
            oos_data="oos",
            kind="ml_model",
        )

    def fit(self, train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        leaks = leakage_check(list(train) + list(validation))
        if leaks:
            return {"fitted": False, "leaks": leaks, "status": "REJECTED_LEAKAGE"}
        keys = sorted({k for row in train for k in dict(row.get("features") or {})})
        weights: dict[str, float] = {}
        for key in keys:
            xs = []
            ys = []
            for row in train:
                try:
                    xs.append(float((row.get("features") or {}).get(key) or 0.0))
                    ys.append(1.0 if row.get("label") else -1.0)
                except (TypeError, ValueError):
                    continue
            if len(xs) < 8:
                continue
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = sum((x - mx) ** 2 for x in xs) or 1.0
            weights[key] = num / den
        self.weights = weights
        val_score = self._expectancy(validation)
        return {
            "fitted": True,
            "weights": dict(weights),
            "validation_expectancy": val_score,
            "leaks": [],
            "can_execute": False,
        }

    def _expectancy(self, rows: Sequence[Mapping[str, Any]]) -> float | None:
        if not rows:
            return None
        pnls = []
        for row in rows:
            score = _dot(self.weights, dict(row.get("features") or {}))
            # Take top half as "selected" for the challenger — research only.
            pnls.append((score, float(row.get("pnl") or 0.0)))
        pnls.sort(reverse=True)
        taken = [p for _, p in pnls[: max(1, len(pnls) // 2)]]
        return round(sum(taken) / len(taken), 6) if taken else None

    def score_oos(self, oos: Sequence[Mapping[str, Any]], *, champion_pnls: Sequence[float]) -> dict[str, Any]:
        leaks = leakage_check(oos)
        if leaks:
            return {"status": "REJECTED_LEAKAGE", "leaks": leaks, "can_execute": False}
        ml = self._expectancy(oos)
        champ = sum(champion_pnls) / len(champion_pnls) if champion_pnls else None
        incremental = None if ml is None or champ is None else round(ml - champ, 6)
        stable = bool(incremental is not None and incremental > 0.05 and len(oos) >= 30)
        contrib = sorted(
            ({"feature": k, "weight": round(v, 6)} for k, v in self.weights.items()),
            key=lambda r: -abs(r["weight"]),
        )
        return {
            "role": ROLE,
            "can_execute": False,
            "status": SHADOW if not stable else SHADOW,  # still shadow until explicit promotion
            "ml_oos_expectancy": ml,
            "champion_oos_expectancy": None if champ is None else round(champ, 6),
            "incremental_oos": incremental,
            "stable_incremental_oos": stable,
            "recommend_promote": False if not stable else False,  # never auto-promote
            "feature_contributions": contrib,
            "n_oos": len(oos),
            "ai_score_written": False,
            "same_execution_reality": True,
            "leaks": [],
        }

    def walk_forward(self, folds: Sequence[Mapping[str, Sequence[Mapping[str, Any]]]]) -> dict[str, Any]:
        results = []
        for fold in folds:
            self.fit(list(fold.get("train") or []), list(fold.get("val") or []))
            results.append(self._expectancy(list(fold.get("oos") or [])))
        measured = [r for r in results if r is not None]
        return {
            "folds": results,
            "mean_oos": None if not measured else round(sum(measured) / len(measured), 6),
            "can_execute": False,
        }
