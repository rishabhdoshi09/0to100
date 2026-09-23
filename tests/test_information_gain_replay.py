from research.autonomy.information_gain import rank_historical_sessions

def req(**kw):
    x={"status":"OPEN","allowed_lanes":["HISTORICAL_REPLAY"],"request_id":"r","strategy_id":"breakout","thesis_hash":"t1","sample_deficit":10,"missing_metrics":["walk_forward_ok"]}; x.update(kw); return x

def cand(date="2026-01-02",**kw):
    x={"session_date":date,"strategy_id":"breakout","thesis_hash":"t1","universe_snapshot_id":"u"+date,"data_version":"d1","feature_version":"f1","model_version":"m1","signal_registry_version":"s1","decision_fingerprint":"fp"+date,"eligible_sample_count":2,"metrics_available":[],"regime_novelty":0,"sector_novelty":0,"decision_uncertainty":0}; x.update(kw); return x

def test_deterministic_and_information_driven():
    low=cand(); high=cand("2026-01-03",eligible_sample_count=5,metrics_available=["walk_forward_ok"],regime_novelty=1,decision_uncertainty=1)
    a=rank_historical_sessions(req(),[low,high]); b=rank_historical_sessions(req(),[high,low])
    assert a==b and a[0]["session_date"]=="2026-01-03" and a[0]["evidence_origin"]=="HISTORICAL_REPLAY" and a[0]["acquisition_fingerprint"].startswith("acq_")
def test_forward_request_is_not_historical(): assert rank_historical_sessions(req(allowed_lanes=["FORWARD_PAPER"]),[cand()])==[]
def test_missing_immutable_identity_rejected(): assert rank_historical_sessions(req(),[cand(universe_snapshot_id="")])==[]
def test_thesis_mismatch_rejected(): assert rank_historical_sessions(req(),[cand(thesis_hash="other")])==[]
def test_closed_request_no_work(): assert rank_historical_sessions(req(status="SATISFIED"),[cand()])==[]
