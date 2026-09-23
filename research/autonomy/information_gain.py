"""Deterministic acquisition ranking for historical replay.

Ranks already-eligible point-in-time historical sessions by the research
evidence gap they can reduce. It cannot create forward evidence or execution
authority.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
SCHEMA_VERSION = 1
EVIDENCE_ORIGIN = "HISTORICAL_REPLAY"
def _stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
@dataclass(frozen=True)
class ReplayAcquisition:
    session_date: str; score: float; rationale: tuple[str, ...]; evidence_origin: str; request_id: str; strategy_id: str; thesis_hash: str; universe_snapshot_id: str; data_version: str; feature_version: str; model_version: str; signal_registry_version: str; decision_fingerprint: str; acquisition_fingerprint: str
    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self); payload["schema_version"] = SCHEMA_VERSION; return payload
def _value(candidate: Mapping[str, Any], key: str) -> float:
    try: return max(0.0, float(candidate.get(key) or 0.0))
    except (TypeError, ValueError): return 0.0
def rank_historical_sessions(request: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if str(request.get("status") or "").upper() != "OPEN": return []
    if EVIDENCE_ORIGIN not in {str(x).upper() for x in request.get("allowed_lanes") or ()}: return []
    deficit=max(0,int(request.get("sample_deficit") or 0)); missing={str(x) for x in request.get("missing_metrics") or ()}; request_id=str(request.get("request_id") or ""); strategy_id=str(request.get("strategy_id") or ""); thesis_hash=str(request.get("thesis_hash") or ""); ranked=[]
    identity_keys=("session_date","universe_snapshot_id","data_version","feature_version","model_version","signal_registry_version","decision_fingerprint")
    for c in candidates:
        if any(not str(c.get(k) or "").strip() for k in identity_keys): continue
        ct=str(c.get("thesis_hash") or ""); cs=str(c.get("strategy_id") or "")
        if thesis_hash and ct != thesis_hash: continue
        if strategy_id and cs and cs != strategy_id: continue
        rationale=[]; score=0.0
        if deficit:
            y=min(float(deficit),_value(c,"eligible_sample_count"))
            if y: score+=y/max(1,deficit); rationale.append(f"sample_deficit:{int(y)}")
        gain=len(missing & {str(x) for x in c.get("metrics_available") or ()})
        if gain: score+=gain; rationale.append(f"missing_metrics:{gain}")
        for key,weight,label in (("regime_novelty",.25,"regime_novelty"),("sector_novelty",.15,"sector_novelty"),("decision_uncertainty",.25,"decision_uncertainty")):
            v=min(1.0,_value(c,key))
            if v: score+=weight*v; rationale.append(label)
        if score<=0: continue
        stable={"request_id":request_id,"session_date":str(c["session_date"])[:10],"strategy_id":strategy_id,"thesis_hash":ct or thesis_hash,"universe_snapshot_id":str(c["universe_snapshot_id"]),"data_version":str(c["data_version"]),"feature_version":str(c["feature_version"]),"model_version":str(c["model_version"]),"signal_registry_version":str(c["signal_registry_version"]),"decision_fingerprint":str(c["decision_fingerprint"])}
        ranked.append(ReplayAcquisition(stable["session_date"],round(score,6),tuple(rationale),EVIDENCE_ORIGIN,request_id,strategy_id,stable["thesis_hash"],stable["universe_snapshot_id"],stable["data_version"],stable["feature_version"],stable["model_version"],stable["signal_registry_version"],stable["decision_fingerprint"],"acq_"+_stable_hash(stable)[:20]))
    ranked.sort(key=lambda x:(-x.score,x.session_date,x.acquisition_fingerprint)); return [x.as_dict() for x in ranked]
