# SEPA-001R historical candidate replay

## valid_vcp_at_pivot
Intent: valid VCP at pivot — eligible if stop ok
- eligible: `True`
- headline: ELIGIBLE — stock + setup + entry
- good_stock / setup / entry: True / True / True
- rejection: []

```json
{
  "as_of_date": "2021-03-16",
  "atr": 1.770528571428581,
  "base_depth_pct": 14.188,
  "base_start_date": "2020-12-30",
  "benchmark_rs": {
    "available": false,
    "benchmark_pct": null,
    "excess_pp": null,
    "label": "UNKNOWN",
    "lookback": 63,
    "note": "Need ~3 months of official stock and Nifty history.",
    "stock_pct": null
  },
  "buy_zone_high": 212.947,
  "buy_zone_low": 209.2755,
  "ca_complete": true,
  "config_hash": "3486d099557347ed",
  "contraction_count": 3,
  "contraction_dates": [
    "2021-01-14",
    "2021-02-08",
    "2021-03-03"
  ],
  "contraction_depths": [
    14.188,
    7.229,
    4.161
  ],
  "contraction_durations": [
    11,
    11,
    11
  ],
  "data_timestamp": "2021-03-16",
  "distance_from_pivot_pct": -0.143,
  "dry_up_ratio": 0.2478,
  "eligibility_version": "sepa-001r.v1",
  "eligible": true,
  "entry_rejection": null,
  "entry_valid": true,
  "evidence": {
    "atr_wide_diagnostic": true,
    "broken_out": false,
    "buy_zone_above_pct": 1.5,
    "near_sepa": false,
    "pit": {
      "ca_complete": true,
      "ca_note": "",
      "pit_class": "PIT_DEGRADED",
      "universe_complete": true,
      "universe_note": ""
    },
    "rs_injected": true,
    "vcp_evidence": {
      "causal": true,
      "far_below_pivot": false,
      "last_contraction_high": 209.8,
      "last_contraction_high_date": "2021-02-16",
      "legacy_too_far_below_would_fail": false,
      "lookback": 120,
      "min_reversal_pct": 2.5,
      "pattern_high": 209.9,
      "pattern_high_date": "2020-12-30",
      "pivot_version": "pivot_last_contraction_v1",
      "raw_contraction_count": 3,
      "swing_highs": 3,
      "swing_lows": 3,
      "vcp_version": "vcp_causal_v1",
      "volume_dry_up_required": true
    },
    "vcp_state": "ENTRY_READY"
  },
  "extended": false,
  "final_contraction_pct": 4.161,
  "good_entry": true,
  "good_setup": true,
  "good_stock": true,
  "headline": "ELIGIBLE \u2014 stock + setup + entry",
  "levels": {
    "above_low_pct": 89.42,
    "below_high_pct": 0.19,
    "high_52w": 209.9,
    "low_52w": 110.6,
    "price": 209.5,
    "sma150": 191.152,
    "sma200": 180.6765,
    "sma200_prev": 172.8248,
    "sma50": 200.1607
  },
  "measured_move": 14.188,
  "pit_class": "PIT_DEGRADED",
  "pit_safe": true,
  "pivot": 209.8,
  "pivot_date": "2021-02-16",
  "pivot_knowable_date": "2021-02-26",
  "pivot_type": "last_contraction_resistance",
  "pivot_version": "pivot_last_contraction_v1",
  "price": 209.5,
  "proposed_entry": 209.5,
  "reasons": [],
  "rejection_codes": [],
  "research_grade": true,
  "resistance": {
    "kind": "measured_move_from_first_contraction",
    "price": 239.5664
  },
  "reward_price": 239.5664,
  "reward_risk": 3.567,
  "reward_status": "MEASURED_MOVE",
  "risk_r": 1.0,
  "rs_components": {},
  "rs_pass": true,
  "rs_percentile": 92.0,
  "rs_score": null,
  "rs_threshold": 70.0,
  "setup_fail_reasons": [],
  "setup_id": "23eadf2fac9e7dd6",
  "setup_quality": 89.59,
  "setup_type": "VCP",
  "stop_atr_multiple": 4.7613,
  "stop_basis": "final_contraction_low",
  "stop_distance_pct": 4.0239,
  "stop_ok": true,
  "structural_stop": 201.07,
  "structure_pass": true,
  "symbol": "VALID_VCP_AT_PIVOT",
  "tightness": 0.2933,
  "trend_passed": 8,
  "trend_rules": [
    {
      "detail": "Close 209.50 vs SMA150 191.15 / SMA200 180.68",
      "id": "price_gt_150_200",
      "passed": true,
      "values": {
        "price": 209.5,
        "sma150": 191.15198316666667,
        "sma200": 180.676487375
      }
    },
    {
      "detail": "SMA150 191.15 vs SMA200 180.68",
      "id": "sma150_gt_200",
      "passed": true,
      "values": {
        "sma150": 191.15198316666667,
        "sma200": 180.676487375
      }
    },
    {
      "detail": "SMA200 180.68 vs 172.82 (21 sessions earlier)",
      "id": "sma200_rising",
      "passed": true,
      "values": {
        "sma200": 180.676487375,
        "sma200_prev": 172.824829875
      }
    },
    {
      "detail": "SMA50 200.16 vs SMA150 191.15 / SMA200 180.68",
      "id": "sma50_leads",
      "passed": true,
      "values": {
        "sma150": 191.15198316666667,
        "sma200": 180.676487375,
        "sma50": 200.16069949999996
      }
    },
    {
      "detail": "Close 209.50 vs SMA50 200.16",
      "id": "price_gt_sma50",
      "passed": true,
      "values": {
        "price": 209.5,
        "sma50": 200.16069949999996
      }
    },
    {
      "detail": "89.4% above 52-week low (need \u226530%)",
      "id": "off_52w_low",
      "passed": true,
      "values": {
        "above_low_pct": 89.42133815551539,
        "low_52w": 110.6,
        "threshold": 30.0
      }
    },
    {
      "detail": "0.2% below 52-week high (need \u226425%)",
      "id": "near_52w_high",
      "passed": true,
      "values": {
        "below_high_pct": 0.1905669366364937,
        "high_52w": 209.9,
        "threshold": 25.0
      }
    },
    {
      "detail": "RS percentile 92.0 (need \u226570)",
      "id": "rs_percentile",
      "passed": true,
      "values": {
        "rs_percentile": 92.0,
        "threshold": 70.0
      }
    }
  ],
  "trend_template_pass": true,
  "trend_total": 8,
  "universe_complete": true,
  "universe_version": "pit_universe",
  "vcp_detected": true,
  "vcp_knowable_date": "2021-03-10",
  "vcp_state": "ENTRY_READY",
  "vcp_version": "vcp_causal_v1",
  "vol_final": 93333.33,
  "vol_first": 376666.67,
  "vol_recent_vs_base": 0.3098
}
```

## stage2_leader_no_vcp
Intent: Stage-2 leader with no VCP → no trade
- eligible: `False`
- headline: GOOD STOCK — SETUP NOT STRUCTURAL
- good_stock / setup / entry: True / False / False
- rejection: ['VCP_NOT_DETECTED', 'NO_PIVOT']

```json
{
  "as_of_date": "2021-01-27",
  "atr": 0.950000000000007,
  "base_depth_pct": null,
  "base_start_date": null,
  "benchmark_rs": {
    "available": false,
    "benchmark_pct": null,
    "excess_pp": null,
    "label": "UNKNOWN",
    "lookback": 63,
    "note": "Need ~3 months of official stock and Nifty history.",
    "stock_pct": null
  },
  "buy_zone_high": null,
  "buy_zone_low": null,
  "ca_complete": true,
  "config_hash": "3486d099557347ed",
  "contraction_count": 0,
  "contraction_dates": [],
  "contraction_depths": [],
  "contraction_durations": [],
  "data_timestamp": "2021-01-27",
  "distance_from_pivot_pct": null,
  "dry_up_ratio": null,
  "eligibility_version": "sepa-001r.v1",
  "eligible": false,
  "entry_rejection": "NO_PIVOT",
  "entry_valid": false,
  "evidence": {
    "atr_wide_diagnostic": false,
    "broken_out": false,
    "buy_zone_above_pct": 1.5,
    "near_sepa": false,
    "pit": {
      "ca_complete": true,
      "ca_note": "",
      "pit_class": "PIT_DEGRADED",
      "universe_complete": true,
      "universe_note": ""
    },
    "rs_injected": true,
    "vcp_evidence": {
      "causal": true,
      "lookback": 120,
      "min_reversal_pct": 2.5,
      "swing_highs": 0,
      "swing_lows": 0,
      "vcp_version": "vcp_causal_v1"
    },
    "vcp_state": "NO_SETUP"
  },
  "extended": false,
  "final_contraction_pct": null,
  "good_entry": false,
  "good_setup": false,
  "good_stock": true,
  "headline": "GOOD STOCK \u2014 SETUP NOT STRUCTURAL",
  "levels": {
    "above_low_pct": 145.74,
    "below_high_pct": 0.17,
    "high_52w": 233.85,
    "low_52w": 95.0,
    "price": 233.45,
    "sma150": 192.475,
    "sma200": 178.725,
    "sma200_prev": 167.175,
    "sma50": 219.975
  },
  "measured_move": null,
  "pit_class": "PIT_DEGRADED",
  "pit_safe": true,
  "pivot": null,
  "pivot_date": null,
  "pivot_knowable_date": null,
  "pivot_type": null,
  "pivot_version": "pivot_last_contraction_v1",
  "price": 233.45000000000002,
  "proposed_entry": null,
  "reasons": [
    "NO_SWING_STRUCTURE",
    "No structurally valid pivot \u2014 none manufactured."
  ],
  "rejection_codes": [
    "VCP_NOT_DETECTED",
    "NO_PIVOT"
  ],
  "research_grade": false,
  "resistance": {},
  "reward_price": null,
  "reward_risk": null,
  "reward_status": "UNKNOWN",
  "risk_r": null,
  "rs_components": {},
  "rs_pass": true,
  "rs_percentile": 92.0,
  "rs_score": null,
  "rs_threshold": 70.0,
  "setup_fail_reasons": [
    "NO_SWING_STRUCTURE"
  ],
  "setup_id": "",
  "setup_quality": null,
  "setup_type": "",
  "stop_atr_multiple": null,
  "stop_basis": null,
  "stop_distance_pct": null,
  "stop_ok": false,
  "structural_stop": null,
  "structure_pass": true,
  "symbol": "STAGE2_LEADER_NO_VCP",
  "tightness": null,
  "trend_passed": 8,
  "trend_rules": [
    {
      "detail": "Close 233.45 vs SMA150 192.47 / SMA200 178.72",
      "id": "price_gt_150_200",
      "passed": true,
      "values": {
        "price": 233.45000000000002,
        "sma150": 192.475,
        "sma200": 178.725
      }
    },
    {
      "detail": "SMA150 192.47 vs SMA200 178.72",
      "id": "sma150_gt_200",
      "passed": true,
      "values": {
        "sma150": 192.475,
        "sma200": 178.725
      }
    },
    {
      "detail": "SMA200 178.72 vs 167.18 (21 sessions earlier)",
      "id": "sma200_rising",
      "passed": true,
      "values": {
        "sma200": 178.725,
        "sma200_prev": 167.175
      }
    },
    {
      "detail": "SMA50 219.97 vs SMA150 192.47 / SMA200 178.72",
      "id": "sma50_leads",
      "passed": true,
      "values": {
        "sma150": 192.475,
        "sma200": 178.725,
        "sma50": 219.975
      }
    },
    {
      "detail": "Close 233.45 vs SMA50 219.97",
      "id": "price_gt_sma50",
      "passed": true,
      "values": {
        "price": 233.45000000000002,
        "sma50": 219.975
      }
    },
    {
      "detail": "145.7% above 52-week low (need \u226530%)",
      "id": "off_52w_low",
      "passed": true,
      "values": {
        "above_low_pct": 145.73684210526318,
        "low_52w": 95.0,
        "threshold": 30.0
      }
    },
    {
      "detail": "0.2% below 52-week high (need \u226425%)",
      "id": "near_52w_high",
      "passed": true,
      "values": {
        "below_high_pct": 0.1710498182595721,
        "high_52w": 233.85000000000002,
        "threshold": 25.0
      }
    },
    {
      "detail": "RS percentile 92.0 (need \u226570)",
      "id": "rs_percentile",
      "passed": true,
      "values": {
        "rs_percentile": 92.0,
        "threshold": 70.0
      }
    }
  ],
  "trend_template_pass": true,
  "trend_total": 8,
  "universe_complete": true,
  "universe_version": "pit_universe",
  "vcp_detected": false,
  "vcp_knowable_date": null,
  "vcp_state": "NO_SETUP",
  "vcp_version": "vcp_causal_v1",
  "vol_final": null,
  "vol_first": null,
  "vol_recent_vs_base": null
}
```

## vcp_weak_rs
Intent: VCP with weak RS → no trade
- eligible: `False`
- headline: NOT STAGE-2 / RS LEADER
- good_stock / setup / entry: False / True / True
- rejection: ['TREND_TEMPLATE_FAIL', 'RS_FAIL']

```json
{
  "as_of_date": "2021-03-16",
  "atr": 1.770528571428581,
  "base_depth_pct": 14.188,
  "base_start_date": "2020-12-30",
  "benchmark_rs": {
    "available": false,
    "benchmark_pct": null,
    "excess_pp": null,
    "label": "UNKNOWN",
    "lookback": 63,
    "note": "Need ~3 months of official stock and Nifty history.",
    "stock_pct": null
  },
  "buy_zone_high": 212.947,
  "buy_zone_low": 209.2755,
  "ca_complete": true,
  "config_hash": "3486d099557347ed",
  "contraction_count": 3,
  "contraction_dates": [
    "2021-01-14",
    "2021-02-08",
    "2021-03-03"
  ],
  "contraction_depths": [
    14.188,
    7.229,
    4.161
  ],
  "contraction_durations": [
    11,
    11,
    11
  ],
  "data_timestamp": "2021-03-16",
  "distance_from_pivot_pct": -0.143,
  "dry_up_ratio": 0.2478,
  "eligibility_version": "sepa-001r.v1",
  "eligible": false,
  "entry_rejection": null,
  "entry_valid": true,
  "evidence": {
    "atr_wide_diagnostic": true,
    "broken_out": false,
    "buy_zone_above_pct": 1.5,
    "near_sepa": true,
    "pit": {
      "ca_complete": true,
      "ca_note": "",
      "pit_class": "PIT_DEGRADED",
      "universe_complete": true,
      "universe_note": ""
    },
    "rs_injected": true,
    "vcp_evidence": {
      "causal": true,
      "far_below_pivot": false,
      "last_contraction_high": 209.8,
      "last_contraction_high_date": "2021-02-16",
      "legacy_too_far_below_would_fail": false,
      "lookback": 120,
      "min_reversal_pct": 2.5,
      "pattern_high": 209.9,
      "pattern_high_date": "2020-12-30",
      "pivot_version": "pivot_last_contraction_v1",
      "raw_contraction_count": 3,
      "swing_highs": 3,
      "swing_lows": 3,
      "vcp_version": "vcp_causal_v1",
      "volume_dry_up_required": true
    },
    "vcp_state": "ENTRY_READY"
  },
  "extended": false,
  "final_contraction_pct": 4.161,
  "good_entry": true,
  "good_setup": true,
  "good_stock": false,
  "headline": "NOT STAGE-2 / RS LEADER",
  "levels": {
    "above_low_pct": 89.42,
    "below_high_pct": 0.19,
    "high_52w": 209.9,
    "low_52w": 110.6,
    "price": 209.5,
    "sma150": 191.152,
    "sma200": 180.6765,
    "sma200_prev": 172.8248,
    "sma50": 200.1607
  },
  "measured_move": 14.188,
  "pit_class": "PIT_DEGRADED",
  "pit_safe": true,
  "pivot": 209.8,
  "pivot_date": "2021-02-16",
  "pivot_knowable_date": "2021-02-26",
  "pivot_type": "last_contraction_resistance",
  "pivot_version": "pivot_last_contraction_v1",
  "price": 209.5,
  "proposed_entry": 209.5,
  "reasons": [
    "Trend template not 8/8: rs_percentile"
  ],
  "rejection_codes": [
    "TREND_TEMPLATE_FAIL",
    "RS_FAIL"
  ],
  "research_grade": false,
  "resistance": {
    "kind": "measured_move_from_first_contraction",
    "price": 239.5664
  },
  "reward_price": 239.5664,
  "reward_risk": 3.567,
  "reward_status": "MEASURED_MOVE",
  "risk_r": 1.0,
  "rs_components": {},
  "rs_pass": false,
  "rs_percentile": 55.0,
  "rs_score": null,
  "rs_threshold": 70.0,
  "setup_fail_reasons": [],
  "setup_id": "3522816a92693540",
  "setup_quality": 89.59,
  "setup_type": "VCP",
  "stop_atr_multiple": 4.7613,
  "stop_basis": "final_contraction_low",
  "stop_distance_pct": 4.0239,
  "stop_ok": true,
  "structural_stop": 201.07,
  "structure_pass": true,
  "symbol": "VCP_WEAK_RS",
  "tightness": 0.2933,
  "trend_passed": 7,
  "trend_rules": [
    {
      "detail": "Close 209.50 vs SMA150 191.15 / SMA200 180.68",
      "id": "price_gt_150_200",
      "passed": true,
      "values": {
        "price": 209.5,
        "sma150": 191.15198316666667,
        "sma200": 180.676487375
      }
    },
    {
      "detail": "SMA150 191.15 vs SMA200 180.68",
      "id": "sma150_gt_200",
      "passed": true,
      "values": {
        "sma150": 191.15198316666667,
        "sma200": 180.676487375
      }
    },
    {
      "detail": "SMA200 180.68 vs 172.82 (21 sessions earlier)",
      "id": "sma200_rising",
      "passed": true,
      "values": {
        "sma200": 180.676487375,
        "sma200_prev": 172.824829875
      }
    },
    {
      "detail": "SMA50 200.16 vs SMA150 191.15 / SMA200 180.68",
      "id": "sma50_leads",
      "passed": true,
      "values": {
        "sma150": 191.15198316666667,
        "sma200": 180.676487375,
        "sma50": 200.16069949999996
      }
    },
    {
      "detail": "Close 209.50 vs SMA50 200.16",
      "id": "price_gt_sma50",
      "passed": true,
      "values": {
        "price": 209.5,
        "sma50": 200.16069949999996
      }
    },
    {
      "detail": "89.4% above 52-week low (need \u226530%)",
      "id": "off_52w_low",
      "passed": true,
      "values": {
        "above_low_pct": 89.42133815551539,
        "low_52w": 110.6,
        "threshold": 30.0
      }
    },
    {
      "detail": "0.2% below 52-week high (need \u226425%)",
      "id": "near_52w_high",
      "passed": true,
      "values": {
        "below_high_pct": 0.1905669366364937,
        "high_52w": 209.9,
        "threshold": 25.0
      }
    },
    {
      "detail": "RS percentile 55.0 (need \u226570)",
      "id": "rs_percentile",
      "passed": false,
      "values": {
        "rs_percentile": 55.0,
        "threshold": 70.0
      }
    }
  ],
  "trend_template_pass": false,
  "trend_total": 8,
  "universe_complete": true,
  "universe_version": "pit_universe",
  "vcp_detected": true,
  "vcp_knowable_date": "2021-03-10",
  "vcp_state": "ENTRY_READY",
  "vcp_version": "vcp_causal_v1",
  "vol_final": 93333.33,
  "vol_first": 376666.67,
  "vol_recent_vs_base": 0.3098
}
```

## vcp_wide_stop
Intent: VCP with wide structural stop → no trade
- eligible: `False`
- headline: NOT STAGE-2 / RS LEADER
- good_stock / setup / entry: False / False / False
- rejection: ['TREND_TEMPLATE_FAIL', 'VCP_NOT_DETECTED', 'WIDE_STRUCTURAL_STOP']

```json
{
  "as_of_date": "2021-03-16",
  "atr": 15.816585714285724,
  "base_depth_pct": 57.106,
  "base_start_date": "2020-12-30",
  "benchmark_rs": {
    "available": false,
    "benchmark_pct": null,
    "excess_pp": null,
    "label": "UNKNOWN",
    "lookback": 63,
    "note": "Need ~3 months of official stock and Nifty history.",
    "stock_pct": null
  },
  "buy_zone_high": 212.947,
  "buy_zone_low": 209.2755,
  "ca_complete": true,
  "config_hash": "3486d099557347ed",
  "contraction_count": 3,
  "contraction_dates": [
    "2021-01-14",
    "2021-02-08",
    "2021-03-03"
  ],
  "contraction_depths": [
    57.106,
    53.627,
    52.092
  ],
  "contraction_durations": [
    11,
    11,
    11
  ],
  "data_timestamp": "2021-03-16",
  "distance_from_pivot_pct": -0.143,
  "dry_up_ratio": 0.2478,
  "eligibility_version": "sepa-001r.v1",
  "eligible": false,
  "entry_rejection": "WIDE_STRUCTURAL_STOP",
  "entry_valid": true,
  "evidence": {
    "atr_wide_diagnostic": true,
    "broken_out": false,
    "buy_zone_above_pct": 1.5,
    "near_sepa": true,
    "pit": {
      "ca_complete": true,
      "ca_note": "",
      "pit_class": "PIT_DEGRADED",
      "universe_complete": true,
      "universe_note": ""
    },
    "rs_injected": true,
    "vcp_evidence": {
      "causal": true,
      "far_below_pivot": false,
      "last_contraction_high": 209.8,
      "last_contraction_high_date": "2021-02-16",
      "legacy_too_far_below_would_fail": false,
      "lookback": 120,
      "min_reversal_pct": 2.5,
      "pattern_high": 209.9,
      "pattern_high_date": "2020-12-30",
      "pivot_version": "pivot_last_contraction_v1",
      "raw_contraction_count": 3,
      "swing_highs": 3,
      "swing_lows": 3,
      "vcp_version": "vcp_causal_v1",
      "volume_dry_up_required": true
    },
    "vcp_state": "FAILED"
  },
  "extended": false,
  "final_contraction_pct": 52.092,
  "good_entry": false,
  "good_setup": false,
  "good_stock": false,
  "headline": "NOT STAGE-2 / RS LEADER",
  "levels": {
    "above_low_pct": 132.69,
    "below_high_pct": 0.19,
    "high_52w": 209.9,
    "low_52w": 90.035,
    "price": 209.5,
    "sma150": 181.5083,
    "sma200": 173.4437,
    "sma200_prev": 168.1262,
    "sma50": 171.2296
  },
  "measured_move": 57.106,
  "pit_class": "PIT_DEGRADED",
  "pit_safe": true,
  "pivot": 209.8,
  "pivot_date": "2021-02-16",
  "pivot_knowable_date": "2021-02-26",
  "pivot_type": "last_contraction_resistance",
  "pivot_version": "pivot_last_contraction_v1",
  "price": 209.5,
  "proposed_entry": 209.5,
  "reasons": [
    "Trend template not 8/8: sma50_leads",
    "NOT_TIGHTENING",
    "FINAL_CONTRACTION_LOOSE",
    "BASE_TOO_DEEP",
    "Structural stop is too far from entry; not tightened artificially."
  ],
  "rejection_codes": [
    "TREND_TEMPLATE_FAIL",
    "VCP_NOT_DETECTED",
    "WIDE_STRUCTURAL_STOP"
  ],
  "research_grade": false,
  "resistance": {
    "kind": "measured_move_from_first_contraction",
    "price": 329.6084
  },
  "reward_price": 329.6084,
  "reward_risk": 1.102,
  "reward_status": "MEASURED_MOVE",
  "risk_r": null,
  "rs_components": {},
  "rs_pass": true,
  "rs_percentile": 92.0,
  "rs_score": null,
  "rs_threshold": 70.0,
  "setup_fail_reasons": [
    "NOT_TIGHTENING",
    "FINAL_CONTRACTION_LOOSE",
    "BASE_TOO_DEEP"
  ],
  "setup_id": "5f54863f8b8b5b69",
  "setup_quality": 50.0,
  "setup_type": "",
  "stop_atr_multiple": 6.8909,
  "stop_basis": "final_contraction_low",
  "stop_distance_pct": 52.0239,
  "stop_ok": false,
  "structural_stop": 100.51,
  "structure_pass": false,
  "symbol": "VCP_WIDE_STOP",
  "tightness": 0.9122,
  "trend_passed": 7,
  "trend_rules": [
    {
      "detail": "Close 209.50 vs SMA150 181.51 / SMA200 173.44",
      "id": "price_gt_150_200",
      "passed": true,
      "values": {
        "price": 209.5,
        "sma150": 181.50827916666665,
        "sma200": 173.443709375
      }
    },
    {
      "detail": "SMA150 181.51 vs SMA200 173.44",
      "id": "sma150_gt_200",
      "passed": true,
      "values": {
        "sma150": 181.50827916666665,
        "sma200": 173.443709375
      }
    },
    {
      "detail": "SMA200 173.44 vs 168.13 (21 sessions earlier)",
      "id": "sma200_rising",
      "passed": true,
      "values": {
        "sma200": 173.443709375,
        "sma200_prev": 168.12616387499997
      }
    },
    {
      "detail": "SMA50 171.23 vs SMA150 181.51 / SMA200 173.44",
      "id": "sma50_leads",
      "passed": false,
      "values": {
        "sma150": 181.50827916666665,
        "sma200": 173.443709375,
        "sma50": 171.22958749999998
      }
    },
    {
      "detail": "Close 209.50 vs SMA50 171.23",
      "id": "price_gt_sma50",
      "passed": true,
      "values": {
        "price": 209.5,
        "sma50": 171.22958749999998
      }
    },
    {
      "detail": "132.7% above 52-week low (need \u226530%)",
      "id": "off_52w_low",
      "passed": true,
      "values": {
        "above_low_pct": 132.68728827678126,
        "low_52w": 90.035,
        "threshold": 30.0
      }
    },
    {
      "detail": "0.2% below 52-week high (need \u226425%)",
      "id": "near_52w_high",
      "passed": true,
      "values": {
        "below_high_pct": 0.1905669366364937,
        "high_52w": 209.9,
        "threshold": 25.0
      }
    },
    {
      "detail": "RS percentile 92.0 (need \u226570)",
      "id": "rs_percentile",
      "passed": true,
      "values": {
        "rs_percentile": 92.0,
        "threshold": 70.0
      }
    }
  ],
  "trend_template_pass": false,
  "trend_total": 8,
  "universe_complete": true,
  "universe_version": "pit_universe",
  "vcp_detected": false,
  "vcp_knowable_date": "2021-03-08",
  "vcp_state": "FAILED",
  "vcp_version": "vcp_causal_v1",
  "vol_final": 93333.33,
  "vol_first": 376666.67,
  "vol_recent_vs_base": 0.3098
}
```

## false_vcp_rejected
Intent: false VCP rejected
- eligible: `False`
- headline: GOOD STOCK — SETUP NOT STRUCTURAL
- good_stock / setup / entry: True / False / False
- rejection: ['VCP_NOT_DETECTED', 'WIDE_STRUCTURAL_STOP']

```json
{
  "as_of_date": "2021-03-16",
  "atr": 6.511214285714295,
  "base_depth_pct": 20.176,
  "base_start_date": "2020-12-30",
  "benchmark_rs": {
    "available": false,
    "benchmark_pct": null,
    "excess_pp": null,
    "label": "UNKNOWN",
    "lookback": 63,
    "note": "Need ~3 months of official stock and Nifty history.",
    "stock_pct": null
  },
  "buy_zone_high": 212.947,
  "buy_zone_low": 209.2755,
  "ca_complete": true,
  "config_hash": "3486d099557347ed",
  "contraction_count": 3,
  "contraction_dates": [
    "2021-01-14",
    "2021-02-08",
    "2021-03-03"
  ],
  "contraction_depths": [
    5.205,
    11.264,
    20.138
  ],
  "contraction_durations": [
    11,
    11,
    11
  ],
  "data_timestamp": "2021-03-16",
  "distance_from_pivot_pct": -0.143,
  "dry_up_ratio": 0.2478,
  "eligibility_version": "sepa-001r.v1",
  "eligible": false,
  "entry_rejection": "WIDE_STRUCTURAL_STOP",
  "entry_valid": true,
  "evidence": {
    "atr_wide_diagnostic": true,
    "broken_out": false,
    "buy_zone_above_pct": 1.5,
    "near_sepa": false,
    "pit": {
      "ca_complete": true,
      "ca_note": "",
      "pit_class": "PIT_DEGRADED",
      "universe_complete": true,
      "universe_note": ""
    },
    "rs_injected": true,
    "vcp_evidence": {
      "causal": true,
      "far_below_pivot": false,
      "last_contraction_high": 209.8,
      "last_contraction_high_date": "2021-02-16",
      "legacy_too_far_below_would_fail": false,
      "lookback": 120,
      "min_reversal_pct": 2.5,
      "pattern_high": 209.9,
      "pattern_high_date": "2020-12-30",
      "pivot_version": "pivot_last_contraction_v1",
      "raw_contraction_count": 3,
      "swing_highs": 3,
      "swing_lows": 3,
      "vcp_version": "vcp_causal_v1",
      "volume_dry_up_required": true
    },
    "vcp_state": "FAILED"
  },
  "extended": false,
  "final_contraction_pct": 20.138,
  "good_entry": false,
  "good_setup": false,
  "good_stock": true,
  "headline": "GOOD STOCK \u2014 SETUP NOT STRUCTURAL",
  "levels": {
    "above_low_pct": 89.42,
    "below_high_pct": 0.19,
    "high_52w": 209.9,
    "low_52w": 110.6,
    "price": 209.5,
    "sma150": 189.8346,
    "sma200": 179.6884,
    "sma200_prev": 173.2739,
    "sma50": 196.2085
  },
  "measured_move": 5.205,
  "pit_class": "PIT_DEGRADED",
  "pit_safe": true,
  "pivot": 209.8,
  "pivot_date": "2021-02-16",
  "pivot_knowable_date": "2021-02-25",
  "pivot_type": "last_contraction_resistance",
  "pivot_version": "pivot_last_contraction_v1",
  "price": 209.5,
  "proposed_entry": 209.5,
  "reasons": [
    "NOT_TIGHTENING",
    "EXPANDING_PULLBACKS",
    "FINAL_CONTRACTION_LOOSE",
    "Structural stop is too far from entry; not tightened artificially."
  ],
  "rejection_codes": [
    "VCP_NOT_DETECTED",
    "WIDE_STRUCTURAL_STOP"
  ],
  "research_grade": false,
  "resistance": {
    "kind": "measured_move_from_first_contraction",
    "price": 220.7201
  },
  "reward_price": 220.7201,
  "reward_risk": 0.267,
  "reward_status": "MEASURED_MOVE",
  "risk_r": null,
  "rs_components": {},
  "rs_pass": true,
  "rs_percentile": 92.0,
  "rs_score": null,
  "rs_threshold": 70.0,
  "setup_fail_reasons": [
    "NOT_TIGHTENING",
    "EXPANDING_PULLBACKS",
    "FINAL_CONTRACTION_LOOSE"
  ],
  "setup_id": "4d27f97b282a1d56",
  "setup_quality": 50.0,
  "setup_type": "",
  "stop_atr_multiple": 6.4427,
  "stop_basis": "final_contraction_low",
  "stop_distance_pct": 20.0239,
  "stop_ok": false,
  "structural_stop": 167.55,
  "structure_pass": true,
  "symbol": "FALSE_VCP_REJECTED",
  "tightness": 3.8691,
  "trend_passed": 8,
  "trend_rules": [
    {
      "detail": "Close 209.50 vs SMA150 189.83 / SMA200 179.69",
      "id": "price_gt_150_200",
      "passed": true,
      "values": {
        "price": 209.5,
        "sma150": 189.83457733333333,
        "sma200": 179.688433
      }
    },
    {
      "detail": "SMA150 189.83 vs SMA200 179.69",
      "id": "sma150_gt_200",
      "passed": true,
      "values": {
        "sma150": 189.83457733333333,
        "sma200": 179.688433
      }
    },
    {
      "detail": "SMA200 179.69 vs 173.27 (21 sessions earlier)",
      "id": "sma200_rising",
      "passed": true,
      "values": {
        "sma200": 179.688433,
        "sma200_prev": 173.2739455
      }
    },
    {
      "detail": "SMA50 196.21 vs SMA150 189.83 / SMA200 179.69",
      "id": "sma50_leads",
      "passed": true,
      "values": {
        "sma150": 189.83457733333333,
        "sma200": 179.688433,
        "sma50": 196.208482
      }
    },
    {
      "detail": "Close 209.50 vs SMA50 196.21",
      "id": "price_gt_sma50",
      "passed": true,
      "values": {
        "price": 209.5,
        "sma50": 196.208482
      }
    },
    {
      "detail": "89.4% above 52-week low (need \u226530%)",
      "id": "off_52w_low",
      "passed": true,
      "values": {
        "above_low_pct": 89.42133815551539,
        "low_52w": 110.6,
        "threshold": 30.0
      }
    },
    {
      "detail": "0.2% below 52-week high (need \u226425%)",
      "id": "near_52w_high",
      "passed": true,
      "values": {
        "below_high_pct": 0.1905669366364937,
        "high_52w": 209.9,
        "threshold": 25.0
      }
    },
    {
      "detail": "RS percentile 92.0 (need \u226570)",
      "id": "rs_percentile",
      "passed": true,
      "values": {
        "rs_percentile": 92.0,
        "threshold": 70.0
      }
    }
  ],
  "trend_template_pass": true,
  "trend_total": 8,
  "universe_complete": true,
  "universe_version": "pit_universe",
  "vcp_detected": false,
  "vcp_knowable_date": "2021-03-09",
  "vcp_state": "FAILED",
  "vcp_version": "vcp_causal_v1",
  "vol_final": 93333.33,
  "vol_first": 376666.67,
  "vol_recent_vs_base": 0.3098
}
```

## extended_no_trade
Intent: extended → NO TRADE — INVALID ENTRY
- eligible: `False`
- headline: NO TRADE — INVALID ENTRY
- good_stock / setup / entry: True / True / False
- rejection: ['ENTRY_EXTENDED']

```json
{
  "as_of_date": "2021-03-16",
  "atr": 2.9676714285714394,
  "base_depth_pct": 14.188,
  "base_start_date": "2020-12-30",
  "benchmark_rs": {
    "available": false,
    "benchmark_pct": null,
    "excess_pp": null,
    "label": "UNKNOWN",
    "lookback": 63,
    "note": "Need ~3 months of official stock and Nifty history.",
    "stock_pct": null
  },
  "buy_zone_high": 212.947,
  "buy_zone_low": 209.2755,
  "ca_complete": true,
  "config_hash": "3486d099557347ed",
  "contraction_count": 3,
  "contraction_dates": [
    "2021-01-14",
    "2021-02-08",
    "2021-03-03"
  ],
  "contraction_depths": [
    14.188,
    7.229,
    4.161
  ],
  "contraction_durations": [
    11,
    11,
    11
  ],
  "data_timestamp": "2021-03-16",
  "distance_from_pivot_pct": 7.8456,
  "dry_up_ratio": 0.2478,
  "eligibility_version": "sepa-001r.v1",
  "eligible": false,
  "entry_rejection": "ENTRY_EXTENDED",
  "entry_valid": false,
  "evidence": {
    "atr_wide_diagnostic": true,
    "broken_out": true,
    "buy_zone_above_pct": 1.5,
    "near_sepa": false,
    "pit": {
      "ca_complete": true,
      "ca_note": "",
      "pit_class": "PIT_DEGRADED",
      "universe_complete": true,
      "universe_note": ""
    },
    "rs_injected": true,
    "vcp_evidence": {
      "causal": true,
      "far_below_pivot": false,
      "last_contraction_high": 209.8,
      "last_contraction_high_date": "2021-02-16",
      "legacy_too_far_below_would_fail": false,
      "lookback": 120,
      "min_reversal_pct": 2.5,
      "pattern_high": 209.9,
      "pattern_high_date": "2020-12-30",
      "pivot_version": "pivot_last_contraction_v1",
      "raw_contraction_count": 3,
      "swing_highs": 3,
      "swing_lows": 3,
      "vcp_version": "vcp_causal_v1",
      "volume_dry_up_required": true
    },
    "vcp_state": "EXTENDED"
  },
  "extended": true,
  "final_contraction_pct": 4.161,
  "good_entry": false,
  "good_setup": true,
  "good_stock": true,
  "headline": "NO TRADE \u2014 INVALID ENTRY",
  "levels": {
    "above_low_pct": 104.58,
    "below_high_pct": 0.13,
    "high_52w": 226.56,
    "low_52w": 110.6,
    "price": 226.26,
    "sma150": 191.7106,
    "sma200": 181.0955,
    "sma200_prev": 172.8248,
    "sma50": 201.8367
  },
  "measured_move": 14.188,
  "pit_class": "PIT_DEGRADED",
  "pit_safe": true,
  "pivot": 209.8,
  "pivot_date": "2021-02-16",
  "pivot_knowable_date": "2021-02-26",
  "pivot_type": "last_contraction_resistance",
  "pivot_version": "pivot_last_contraction_v1",
  "price": 226.26000000000002,
  "proposed_entry": 226.26,
  "reasons": [
    "NO TRADE \u2014 INVALID ENTRY: price 226.26000000000002 vs pivot 209.8 (7.8456% above)."
  ],
  "rejection_codes": [
    "ENTRY_EXTENDED"
  ],
  "research_grade": false,
  "resistance": {
    "kind": "measured_move_from_first_contraction",
    "price": 239.5664
  },
  "reward_price": 239.5664,
  "reward_risk": 0.528,
  "reward_status": "MEASURED_MOVE",
  "risk_r": null,
  "rs_components": {},
  "rs_pass": true,
  "rs_percentile": 97.0,
  "rs_score": null,
  "rs_threshold": 70.0,
  "setup_fail_reasons": [],
  "setup_id": "dd42a140ea8249ce",
  "setup_quality": 89.59,
  "setup_type": "VCP",
  "stop_atr_multiple": 8.4881,
  "stop_basis": "final_contraction_low",
  "stop_distance_pct": 11.1332,
  "stop_ok": false,
  "structural_stop": 201.07,
  "structure_pass": true,
  "symbol": "EXTENDED_NO_TRADE",
  "tightness": 0.2933,
  "trend_passed": 8,
  "trend_rules": [
    {
      "detail": "Close 226.26 vs SMA150 191.71 / SMA200 181.10",
      "id": "price_gt_150_200",
      "passed": true,
      "values": {
        "price": 226.26000000000002,
        "sma150": 191.71064983333335,
        "sma200": 181.095487375
      }
    },
    {
      "detail": "SMA150 191.71 vs SMA200 181.10",
      "id": "sma150_gt_200",
      "passed": true,
      "values": {
        "sma150": 191.71064983333335,
        "sma200": 181.095487375
      }
    },
    {
      "detail": "SMA200 181.10 vs 172.82 (21 sessions earlier)",
      "id": "sma200_rising",
      "passed": true,
      "values": {
        "sma200": 181.095487375,
        "sma200_prev": 172.824829875
      }
    },
    {
      "detail": "SMA50 201.84 vs SMA150 191.71 / SMA200 181.10",
      "id": "sma50_leads",
      "passed": true,
      "values": {
        "sma150": 191.71064983333335,
        "sma200": 181.095487375,
        "sma50": 201.83669950000004
      }
    },
    {
      "detail": "Close 226.26 vs SMA50 201.84",
      "id": "price_gt_sma50",
      "passed": true,
      "values": {
        "price": 226.26000000000002,
        "sma50": 201.83669950000004
      }
    },
    {
      "detail": "104.6% above 52-week low (need \u226530%)",
      "id": "off_52w_low",
      "passed": true,
      "values": {
        "above_low_pct": 104.57504520795662,
        "low_52w": 110.6,
        "threshold": 30.0
      }
    },
    {
      "detail": "0.1% below 52-week high (need \u226425%)",
      "id": "near_52w_high",
      "passed": true,
      "values": {
        "below_high_pct": 0.13241525423729472,
        "high_52w": 226.56000000000003,
        "threshold": 25.0
      }
    },
    {
      "detail": "RS percentile 97.0 (need \u226570)",
      "id": "rs_percentile",
      "passed": true,
      "values": {
        "rs_percentile": 97.0,
        "threshold": 70.0
      }
    }
  ],
  "trend_template_pass": true,
  "trend_total": 8,
  "universe_complete": true,
  "universe_version": "pit_universe",
  "vcp_detected": true,
  "vcp_knowable_date": "2021-03-09",
  "vcp_state": "EXTENDED",
  "vcp_version": "vcp_causal_v1",
  "vol_final": 93333.33,
  "vol_first": 376666.67,
  "vol_recent_vs_base": 0.3098
}
```

## below_pivot_not_yet
Intent: below buy-zone — setup may be forming
- eligible: `False`
- headline: NO TRADE — INVALID ENTRY
- good_stock / setup / entry: True / True / False
- rejection: ['ENTRY_BELOW_PIVOT']

```json
{
  "as_of_date": "2021-03-16",
  "atr": 1.4712428571428668,
  "base_depth_pct": 14.188,
  "base_start_date": "2020-12-30",
  "benchmark_rs": {
    "available": false,
    "benchmark_pct": null,
    "excess_pp": null,
    "label": "UNKNOWN",
    "lookback": 63,
    "note": "Need ~3 months of official stock and Nifty history.",
    "stock_pct": null
  },
  "buy_zone_high": 210.8206,
  "buy_zone_low": 207.1857,
  "ca_complete": true,
  "config_hash": "3486d099557347ed",
  "contraction_count": 2,
  "contraction_dates": [
    "2021-01-14",
    "2021-02-08"
  ],
  "contraction_depths": [
    14.188,
    7.229
  ],
  "contraction_durations": [
    11,
    11
  ],
  "data_timestamp": "2021-03-16",
  "distance_from_pivot_pct": -1.1531,
  "dry_up_ratio": 0.6726,
  "eligibility_version": "sepa-001r.v1",
  "eligible": false,
  "entry_rejection": "ENTRY_BELOW_PIVOT",
  "entry_valid": false,
  "evidence": {
    "atr_wide_diagnostic": true,
    "broken_out": false,
    "buy_zone_above_pct": 1.5,
    "near_sepa": false,
    "pit": {
      "ca_complete": true,
      "ca_note": "",
      "pit_class": "PIT_DEGRADED",
      "universe_complete": true,
      "universe_note": ""
    },
    "rs_injected": true,
    "vcp_evidence": {
      "causal": true,
      "far_below_pivot": false,
      "last_contraction_high": 207.705,
      "last_contraction_high_date": "2021-01-22",
      "legacy_too_far_below_would_fail": false,
      "lookback": 120,
      "min_reversal_pct": 2.5,
      "pattern_high": 209.9,
      "pattern_high_date": "2020-12-30",
      "pivot_version": "pivot_last_contraction_v1",
      "raw_contraction_count": 2,
      "swing_highs": 3,
      "swing_lows": 2,
      "vcp_version": "vcp_causal_v1",
      "volume_dry_up_required": true
    },
    "vcp_state": "PIVOT_DEFINED"
  },
  "extended": false,
  "final_contraction_pct": 7.229,
  "good_entry": false,
  "good_setup": true,
  "good_stock": true,
  "headline": "NO TRADE \u2014 INVALID ENTRY",
  "levels": {
    "above_low_pct": 85.63,
    "below_high_pct": 2.19,
    "high_52w": 209.9,
    "low_52w": 110.6,
    "price": 205.31,
    "sma150": 191.0123,
    "sma200": 180.5717,
    "sma200_prev": 172.8248,
    "sma50": 199.7417
  },
  "measured_move": 14.188,
  "pit_class": "PIT_DEGRADED",
  "pit_safe": true,
  "pivot": 207.705,
  "pivot_date": "2021-01-22",
  "pivot_knowable_date": "2021-02-02",
  "pivot_type": "last_contraction_resistance",
  "pivot_version": "pivot_last_contraction_v1",
  "price": 205.31,
  "proposed_entry": 205.31,
  "reasons": [
    "Price still below the buy-zone \u2014 setup may be forming, not a trade."
  ],
  "rejection_codes": [
    "ENTRY_BELOW_PIVOT"
  ],
  "research_grade": false,
  "resistance": {
    "kind": "measured_move_from_first_contraction",
    "price": 237.1742
  },
  "reward_price": 237.1742,
  "reward_risk": 2.525,
  "reward_status": "MEASURED_MOVE",
  "risk_r": 1.0,
  "rs_components": {},
  "rs_pass": true,
  "rs_percentile": 92.0,
  "rs_score": null,
  "rs_threshold": 70.0,
  "setup_fail_reasons": [],
  "setup_id": "30ffe54ee5d26d06",
  "setup_quality": 78.18,
  "setup_type": "VCP",
  "stop_atr_multiple": 8.5778,
  "stop_basis": "final_contraction_low",
  "stop_distance_pct": 6.1468,
  "stop_ok": true,
  "structural_stop": 192.69,
  "structure_pass": true,
  "symbol": "BELOW_PIVOT_NOT_YET",
  "tightness": 0.5095,
  "trend_passed": 8,
  "trend_rules": [
    {
      "detail": "Close 205.31 vs SMA150 191.01 / SMA200 180.57",
      "id": "price_gt_150_200",
      "passed": true,
      "values": {
        "price": 205.31,
        "sma150": 191.01231649999997,
        "sma200": 180.571737375
      }
    },
    {
      "detail": "SMA150 191.01 vs SMA200 180.57",
      "id": "sma150_gt_200",
      "passed": true,
      "values": {
        "sma150": 191.01231649999997,
        "sma200": 180.571737375
      }
    },
    {
      "detail": "SMA200 180.57 vs 172.82 (21 sessions earlier)",
      "id": "sma200_rising",
      "passed": true,
      "values": {
        "sma200": 180.571737375,
        "sma200_prev": 172.824829875
      }
    },
    {
      "detail": "SMA50 199.74 vs SMA150 191.01 / SMA200 180.57",
      "id": "sma50_leads",
      "passed": true,
      "values": {
        "sma150": 191.01231649999997,
        "sma200": 180.571737375,
        "sma50": 199.74169949999995
      }
    },
    {
      "detail": "Close 205.31 vs SMA50 199.74",
      "id": "price_gt_sma50",
      "passed": true,
      "values": {
        "price": 205.31,
        "sma50": 199.74169949999995
      }
    },
    {
      "detail": "85.6% above 52-week low (need \u226530%)",
      "id": "off_52w_low",
      "passed": true,
      "values": {
        "above_low_pct": 85.63291139240508,
        "low_52w": 110.6,
        "threshold": 30.0
      }
    },
    {
      "detail": "2.2% below 52-week high (need \u226425%)",
      "id": "near_52w_high",
      "passed": true,
      "values": {
        "below_high_pct": 2.1867555979037667,
        "high_52w": 209.9,
        "threshold": 25.0
      }
    },
    {
      "detail": "RS percentile 92.0 (need \u226570)",
      "id": "rs_percentile",
      "passed": true,
      "values": {
        "rs_percentile": 92.0,
        "threshold": 70.0
      }
    }
  ],
  "trend_template_pass": true,
  "trend_total": 8,
  "universe_complete": true,
  "universe_version": "pit_universe",
  "vcp_detected": true,
  "vcp_knowable_date": "2021-02-12",
  "vcp_state": "PIVOT_DEFINED",
  "vcp_version": "vcp_causal_v1",
  "vol_final": 253333.33,
  "vol_first": 376666.67,
  "vol_recent_vs_base": 0.2427
}
```

## gap_through_no_trade
Intent: valid VCP that gaps beyond buy-zone → no trade
- eligible: `True`
- headline: ELIGIBLE — stock + setup + entry
- good_stock / setup / entry: True / True / True
- rejection: []

```json
{
  "as_of_date": "2021-03-16",
  "atr": 1.770528571428581,
  "base_depth_pct": 14.188,
  "base_start_date": "2020-12-30",
  "benchmark_rs": {
    "available": false,
    "benchmark_pct": null,
    "excess_pp": null,
    "label": "UNKNOWN",
    "lookback": 63,
    "note": "Need ~3 months of official stock and Nifty history.",
    "stock_pct": null
  },
  "buy_zone_high": 212.947,
  "buy_zone_low": 209.2755,
  "ca_complete": true,
  "config_hash": "3486d099557347ed",
  "contraction_count": 3,
  "contraction_dates": [
    "2021-01-14",
    "2021-02-08",
    "2021-03-03"
  ],
  "contraction_depths": [
    14.188,
    7.229,
    4.161
  ],
  "contraction_durations": [
    11,
    11,
    11
  ],
  "data_timestamp": "2021-03-16",
  "distance_from_pivot_pct": -0.143,
  "dry_up_ratio": 0.2478,
  "eligibility_version": "sepa-001r.v1",
  "eligible": true,
  "entry_rejection": null,
  "entry_valid": true,
  "evidence": {
    "atr_wide_diagnostic": true,
    "broken_out": false,
    "buy_zone_above_pct": 1.5,
    "near_sepa": false,
    "pit": {
      "ca_complete": true,
      "ca_note": "",
      "pit_class": "PIT_DEGRADED",
      "universe_complete": true,
      "universe_note": ""
    },
    "rs_injected": true,
    "vcp_evidence": {
      "causal": true,
      "far_below_pivot": false,
      "last_contraction_high": 209.8,
      "last_contraction_high_date": "2021-02-16",
      "legacy_too_far_below_would_fail": false,
      "lookback": 120,
      "min_reversal_pct": 2.5,
      "pattern_high": 209.9,
      "pattern_high_date": "2020-12-30",
      "pivot_version": "pivot_last_contraction_v1",
      "raw_contraction_count": 3,
      "swing_highs": 3,
      "swing_lows": 3,
      "vcp_version": "vcp_causal_v1",
      "volume_dry_up_required": true
    },
    "vcp_state": "ENTRY_READY"
  },
  "extended": false,
  "final_contraction_pct": 4.161,
  "good_entry": true,
  "good_setup": true,
  "good_stock": true,
  "headline": "ELIGIBLE \u2014 stock + setup + entry",
  "levels": {
    "above_low_pct": 89.42,
    "below_high_pct": 0.19,
    "high_52w": 209.9,
    "low_52w": 110.6,
    "price": 209.5,
    "sma150": 191.152,
    "sma200": 180.6765,
    "sma200_prev": 172.8248,
    "sma50": 200.1607
  },
  "measured_move": 14.188,
  "pit_class": "PIT_DEGRADED",
  "pit_safe": true,
  "pivot": 209.8,
  "pivot_date": "2021-02-16",
  "pivot_knowable_date": "2021-02-26",
  "pivot_type": "last_contraction_resistance",
  "pivot_version": "pivot_last_contraction_v1",
  "price": 209.5,
  "proposed_entry": 209.5,
  "reasons": [],
  "rejection_codes": [],
  "research_grade": true,
  "resistance": {
    "kind": "measured_move_from_first_contraction",
    "price": 239.5664
  },
  "reward_price": 239.5664,
  "reward_risk": 3.567,
  "reward_status": "MEASURED_MOVE",
  "risk_r": 1.0,
  "rs_components": {},
  "rs_pass": true,
  "rs_percentile": 92.0,
  "rs_score": null,
  "rs_threshold": 70.0,
  "setup_fail_reasons": [],
  "setup_id": "66e4978e26929280",
  "setup_quality": 89.59,
  "setup_type": "VCP",
  "stop_atr_multiple": 4.7613,
  "stop_basis": "final_contraction_low",
  "stop_distance_pct": 4.0239,
  "stop_ok": true,
  "structural_stop": 201.07,
  "structure_pass": true,
  "symbol": "GAP",
  "tightness": 0.2933,
  "trend_passed": 8,
  "trend_rules": [
    {
      "detail": "Close 209.50 vs SMA150 191.15 / SMA200 180.68",
      "id": "price_gt_150_200",
      "passed": true,
      "values": {
        "price": 209.5,
        "sma150": 191.15198316666667,
        "sma200": 180.676487375
      }
    },
    {
      "detail": "SMA150 191.15 vs SMA200 180.68",
      "id": "sma150_gt_200",
      "passed": true,
      "values": {
        "sma150": 191.15198316666667,
        "sma200": 180.676487375
      }
    },
    {
      "detail": "SMA200 180.68 vs 172.82 (21 sessions earlier)",
      "id": "sma200_rising",
      "passed": true,
      "values": {
        "sma200": 180.676487375,
        "sma200_prev": 172.824829875
      }
    },
    {
      "detail": "SMA50 200.16 vs SMA150 191.15 / SMA200 180.68",
      "id": "sma50_leads",
      "passed": true,
      "values": {
        "sma150": 191.15198316666667,
        "sma200": 180.676487375,
        "sma50": 200.16069949999996
      }
    },
    {
      "detail": "Close 209.50 vs SMA50 200.16",
      "id": "price_gt_sma50",
      "passed": true,
      "values": {
        "price": 209.5,
        "sma50": 200.16069949999996
      }
    },
    {
      "detail": "89.4% above 52-week low (need \u226530%)",
      "id": "off_52w_low",
      "passed": true,
      "values": {
        "above_low_pct": 89.42133815551539,
        "low_52w": 110.6,
        "threshold": 30.0
      }
    },
    {
      "detail": "0.2% below 52-week high (need \u226425%)",
      "id": "near_52w_high",
      "passed": true,
      "values": {
        "below_high_pct": 0.1905669366364937,
        "high_52w": 209.9,
        "threshold": 25.0
      }
    },
    {
      "detail": "RS percentile 92.0 (need \u226570)",
      "id": "rs_percentile",
      "passed": true,
      "values": {
        "rs_percentile": 92.0,
        "threshold": 70.0
      }
    }
  ],
  "trend_template_pass": true,
  "trend_total": 8,
  "universe_complete": true,
  "universe_version": "pit_universe",
  "vcp_detected": true,
  "vcp_knowable_date": "2021-03-10",
  "vcp_state": "ENTRY_READY",
  "vcp_version": "vcp_causal_v1",
  "vol_final": 93333.33,
  "vol_first": 376666.67,
  "vol_recent_vs_base": 0.3098
}
```

## failed_breakout_contract
Intent: failed breakout is a filled trade that stops / closes back through pivot — not a late chase
- eligible: `None`
- headline: None
- good_stock / setup / entry: None / None / None
- rejection: None

```json
{
  "note": "See ablation failed_break flag on filled E/F rows."
}
```

## successful_2r_contract
Intent: +2R measured on structural risk after a VALID_FILL only
- eligible: `None`
- headline: None
- good_stock / setup / entry: None / None / None
- rejection: None

```json
{
  "note": "See ablation pct_2r on filled E/F rows. No 4\u00d7ATR target."
}
```

## valid_vcp_pre_breakout
Intent: valid VCP detected pre-breakout (prefix of planted coil)
- eligible: `True`
- headline: ELIGIBLE — stock + setup + entry
- good_stock / setup / entry: True / True / True
- rejection: []

```json
{
  "as_of_date": "2021-03-11",
  "atr": 2.0769571428571516,
  "base_depth_pct": 14.188,
  "base_start_date": "2020-12-30",
  "benchmark_rs": {
    "available": false,
    "benchmark_pct": null,
    "excess_pp": null,
    "label": "UNKNOWN",
    "lookback": 63,
    "note": "Need ~3 months of official stock and Nifty history.",
    "stock_pct": null
  },
  "buy_zone_high": 212.947,
  "buy_zone_low": 209.2755,
  "ca_complete": true,
  "config_hash": "3486d099557347ed",
  "contraction_count": 3,
  "contraction_dates": [
    "2021-01-14",
    "2021-02-08",
    "2021-03-03"
  ],
  "contraction_depths": [
    14.188,
    7.229,
    4.161
  ],
  "contraction_durations": [
    11,
    11,
    11
  ],
  "data_timestamp": "2021-03-11",
  "distance_from_pivot_pct": -0.143,
  "dry_up_ratio": 0.2478,
  "eligibility_version": "sepa-001r.v1",
  "eligible": true,
  "entry_rejection": null,
  "entry_valid": true,
  "evidence": {
    "atr_wide_diagnostic": true,
    "broken_out": false,
    "buy_zone_above_pct": 1.5,
    "near_sepa": false,
    "pit": {
      "ca_complete": true,
      "ca_note": "",
      "pit_class": "PIT_DEGRADED",
      "universe_complete": true,
      "universe_note": ""
    },
    "rs_injected": true,
    "vcp_evidence": {
      "causal": true,
      "far_below_pivot": false,
      "last_contraction_high": 209.8,
      "last_contraction_high_date": "2021-02-16",
      "legacy_too_far_below_would_fail": false,
      "lookback": 120,
      "min_reversal_pct": 2.5,
      "pattern_high": 209.9,
      "pattern_high_date": "2020-12-30",
      "pivot_version": "pivot_last_contraction_v1",
      "raw_contraction_count": 3,
      "swing_highs": 3,
      "swing_lows": 3,
      "vcp_version": "vcp_causal_v1",
      "volume_dry_up_required": true
    },
    "vcp_state": "ENTRY_READY"
  },
  "extended": false,
  "final_contraction_pct": 4.161,
  "good_entry": true,
  "good_setup": true,
  "good_stock": true,
  "headline": "ELIGIBLE \u2014 stock + setup + entry",
  "levels": {
    "above_low_pct": 92.03,
    "below_high_pct": 0.19,
    "high_52w": 209.9,
    "low_52w": 109.1,
    "price": 209.5,
    "sma150": 190.182,
    "sma200": 179.574,
    "sma200_prev": 171.7132,
    "sma50": 200.1188
  },
  "measured_move": 14.188,
  "pit_class": "PIT_DEGRADED",
  "pit_safe": true,
  "pivot": 209.8,
  "pivot_date": "2021-02-16",
  "pivot_knowable_date": "2021-02-26",
  "pivot_type": "last_contraction_resistance",
  "pivot_version": "pivot_last_contraction_v1",
  "price": 209.5,
  "proposed_entry": 209.5,
  "reasons": [],
  "rejection_codes": [],
  "research_grade": true,
  "resistance": {
    "kind": "measured_move_from_first_contraction",
    "price": 239.5664
  },
  "reward_price": 239.5664,
  "reward_risk": 3.567,
  "reward_status": "MEASURED_MOVE",
  "risk_r": 1.0,
  "rs_components": {},
  "rs_pass": true,
  "rs_percentile": 92.0,
  "rs_score": null,
  "rs_threshold": 70.0,
  "setup_fail_reasons": [],
  "setup_id": "b761f4ce2e00307d",
  "setup_quality": 89.59,
  "setup_type": "VCP",
  "stop_atr_multiple": 4.0588,
  "stop_basis": "final_contraction_low",
  "stop_distance_pct": 4.0239,
  "stop_ok": true,
  "structural_stop": 201.07,
  "structure_pass": true,
  "symbol": "VALID_VCP_PRE_BREAKOUT",
  "tightness": 0.2933,
  "trend_passed": 8,
  "trend_rules": [
    {
      "detail": "Close 209.50 vs SMA150 190.18 / SMA200 179.57",
      "id": "price_gt_150_200",
      "passed": true,
      "values": {
        "price": 209.5,
        "sma150": 190.18198316666667,
        "sma200": 179.573987375
      }
    },
    {
      "detail": "SMA150 190.18 vs SMA200 179.57",
      "id": "sma150_gt_200",
      "passed": true,
      "values": {
        "sma150": 190.18198316666667,
        "sma200": 179.573987375
      }
    },
    {
      "detail": "SMA200 179.57 vs 171.71 (21 sessions earlier)",
      "id": "sma200_rising",
      "passed": true,
      "values": {
        "sma200": 179.573987375,
        "sma200_prev": 171.713155875
      }
    },
    {
      "detail": "SMA50 200.12 vs SMA150 190.18 / SMA200 179.57",
      "id": "sma50_leads",
      "passed": true,
      "values": {
        "sma150": 190.18198316666667,
        "sma200": 179.573987375,
        "sma50": 200.1187995
      }
    },
    {
      "detail": "Close 209.50 vs SMA50 200.12",
      "id": "price_gt_sma50",
      "passed": true,
      "values": {
        "price": 209.5,
        "sma50": 200.1187995
      }
    },
    {
      "detail": "92.0% above 52-week low (need \u226530%)",
      "id": "off_52w_low",
      "passed": true,
      "values": {
        "above_low_pct": 92.02566452795602,
        "low_52w": 109.1,
        "threshold": 30.0
      }
    },
    {
      "detail": "0.2% below 52-week high (need \u226425%)",
      "id": "near_52w_high",
      "passed": true,
      "values": {
        "below_high_pct": 0.1905669366364937,
        "high_52w": 209.9,
        "threshold": 25.0
      }
    },
    {
      "detail": "RS percentile 92.0 (need \u226570)",
      "id": "rs_percentile",
      "passed": true,
      "values": {
        "rs_percentile": 92.0,
        "threshold": 70.0
      }
    }
  ],
  "trend_template_pass": true,
  "trend_total": 8,
  "universe_complete": true,
  "universe_version": "pit_universe",
  "vcp_detected": true,
  "vcp_knowable_date": "2021-03-10",
  "vcp_state": "ENTRY_READY",
  "vcp_version": "vcp_causal_v1",
  "vol_final": 93333.33,
  "vol_first": 376666.67,
  "vol_recent_vs_base": 0.3098
}
```

## Fill classifications (causal)

- gap_through_no_trade: `{'class': 'GAP_THROUGH', 'fill': None, 'reason': 'next open gapped through buy-zone'}`

## NSE as-of replays (official bhav, as-of slice only)

### CHENNPETRO @ 2026-03-20
Intent: valid VCP at pivot / ENTRY_READY
- eligible: `True`
- headline: ELIGIBLE — stock + setup + entry
- vcp_state: `ENTRY_READY` dist=1.236% pivot=1051.75 stop=1022.0 RS=75.0
- rejection: []
- pit_class: `PIT_DEGRADED` ca_complete=`False`

### LAURUSLABS @ 2026-03-10
Intent: valid VCP at pivot
- eligible: `True`
- headline: ELIGIBLE — stock + setup + entry
- vcp_state: `ENTRY_READY` dist=-0.2115% pivot=1040.2 stop=1001.6 RS=75.0
- rejection: []
- pit_class: `PIT_DEGRADED` ca_complete=`False`

### MOTHERSON @ 2026-04-16
Intent: valid VCP at pivot (+2R candidate path)
- eligible: `False`
- headline: NOT STAGE-2 / RS LEADER
- vcp_state: `ENTRY_READY` dist=0.0163% pivot=122.8 stop=118.4 RS=25.0
- rejection: ['TREND_TEMPLATE_FAIL', 'RS_FAIL']
- pit_class: `PIT_DEGRADED` ca_complete=`False`

### SBIN @ 2026-01-13
Intent: valid VCP at pivot
- eligible: `False`
- headline: NOT STAGE-2 / RS LEADER
- vcp_state: `ENTRY_READY` dist=0.4346% pivot=1024.0 stop=994.0 RS=50.0
- rejection: ['TREND_TEMPLATE_FAIL', 'RS_FAIL']
- pit_class: `PIT_DEGRADED` ca_complete=`False`

### TCS @ 2025-11-14
Intent: new detector coil, often below zone / not a chase
- eligible: `False` headline: NOT STAGE-2 / RS LEADER state=`PIVOT_DEFINED` dist=-0.7668
- rejection: ['TREND_TEMPLATE_FAIL', 'RS_UNAVAILABLE', 'ENTRY_BELOW_PIVOT']

Research-book RS for these names (100-name daily ablation) is in `setups.jsonl`. The tiny replay universe above is only for as-of slicing; it is not the cross-sectional rank used in A–F.
