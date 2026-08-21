# SEPA-001 historical candidate replay

## Synthetic high-quality VCP near pivot
Intent: eligible True
- eligible: `True`
- headline: ELIGIBLE — stock + setup + entry
- good_stock / setup / entry: True / True / True
- rejection: []

```json
{
  "as_of_date": "2021-03-16",
  "atr": 1.8303857142857234,
  "base_depth_pct": 14.188,
  "benchmark_rs": {
    "available": false,
    "benchmark_pct": null,
    "excess_pp": null,
    "label": "UNKNOWN",
    "lookback": 63,
    "note": "Need ~3 months of official stock and Nifty history.",
    "stock_pct": null
  },
  "buy_zone_high": 213.0485,
  "buy_zone_low": 209.3753,
  "ca_complete": true,
  "config_hash": "a1b0b5651651c768",
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
  "distance_from_pivot_pct": 0.2087,
  "dry_up_ratio": 0.2478,
  "eligibility_version": "sepa-001.v1",
  "eligible": true,
  "entry_rejection": null,
  "entry_valid": true,
  "evidence": {
    "buy_zone_above_pct": 1.5,
    "near_sepa": false,
    "pit": {
      "ca_complete": true,
      "ca_note": "",
      "universe_complete": true,
      "universe_note": ""
    },
    "rs_injected": true,
    "vcp_evidence": {
      "lookback": 120,
      "min_reversal_pct": 2.5,
      "raw_contraction_count": 3,
      "swing_highs": 3,
      "swing_lows": 3
    }
  },
  "extended": false,
  "final_contraction_pct": 4.161,
  "good_entry": true,
  "good_setup": true,
  "good_stock": true,
  "headline": "ELIGIBLE \u2014 stock + setup + entry",
  "levels": {
    "above_low_pct": 90.18,
    "below_high_pct": 0.14,
    "high_52w": 210.638,
    "low_52w": 110.6,
    "price": 210.338,
    "sma150": 191.1799,
    "sma200": 180.6974,
    "sma200_prev": 172.8248,
    "sma50": 200.2445
  },
  "measured_move": 14.188,
  "pit_safe": true,
  "pivot": 209.9,
  "pivot_date": "2020-12-30",
  "pivot_type": "vcp_resistance_swing_high",
  "price": 210.338,
  "proposed_entry": 210.338,
  "reasons": [],
  "rejection_codes": [],
  "research_grade": true,
  "resistance": {
    "kind": "measured_move_from_first_contraction",
    "price": 239.6806
  },
  "reward_price": 239.6806,
  "reward_risk": 3.166,
  "reward_status": "MEASURED_MOVE",
  "risk_r": 1.0,
  "rs_components": {},
  "rs_pass": true,
  "rs_percentile": 94.0,
  "rs_score": null,
  "rs_threshold": 70.0,
  "setup_fail_reasons": [],
  "setup_quality": 89.59,
  "setup_type": "VCP",
  "stop_atr_multiple": 5.0634,
  "stop_basis": "final_contraction_low",
  "stop_distance_pct": 4.4062,
  "stop_ok": true,
  "structural_stop": 201.07,
  "structure_pass": true,
  "symbol": "PASS",
  "tightness": 0.2933,
  "trend_passed": 8,
  "trend_rules": [
    {
      "detail": "Close 210.34 vs SMA150 191.18 / SMA200 180.70",
      "id": "price_gt_150_200",
      "passed": true,
      "values": {
        "price": 210.338,
        "sma150": 191.1799165,
        "sma200": 180.69743737500002
      }
    },
    {
      "detail": "SMA150 191.18 vs SMA200 180.70",
      "id": "sma150_gt_200",
      "passed": true,
      "values": {
        "sma150": 191.1799165,
        "sma200": 180.69743737500002
      }
    },
    {
      "detail": "SMA200 180.70 vs 172.82 (21 sessions earlier)",
      "id": "sma200_rising",
      "passed": true,
      "values": {
        "sma200": 180.69743737500002,
        "sma200_prev": 172.824829875
      }
    },
    {
      "detail": "SMA50 200.24 vs SMA150 191.18 / SMA200 180.70",
      "id": "sma50_leads",
      "passed": true,
      "values": {
        "sma150": 191.1799165,
        "sma200": 180.69743737500002,
        "sma50": 200.24449950000002
      }
    },
    {
      "detail": "Close 210.34 vs SMA50 200.24",
      "id": "price_gt_sma50",
      "passed": true,
      "values": {
        "price": 210.338,
        "sma50": 200.24449950000002
      }
    },
    {
      "detail": "90.2% above 52-week low (need \u226530%)",
      "id": "off_52w_low",
      "passed": true,
      "values": {
        "above_low_pct": 90.17902350813745,
        "low_52w": 110.6,
        "threshold": 30.0
      }
    },
    {
      "detail": "0.1% below 52-week high (need \u226425%)",
      "id": "near_52w_high",
      "passed": true,
      "values": {
        "below_high_pct": 0.14242444383255615,
        "high_52w": 210.638,
        "threshold": 25.0
      }
    },
    {
      "detail": "RS percentile 94.0 (need \u226570)",
      "id": "rs_percentile",
      "passed": true,
      "values": {
        "rs_percentile": 94.0,
        "threshold": 70.0
      }
    }
  ],
  "trend_template_pass": true,
  "trend_total": 8,
  "universe_complete": true,
  "universe_version": "pit_universe",
  "vcp_detected": true,
  "vol_final": 93333.33,
  "vol_first": 376666.67,
  "vol_recent_vs_base": 0.3098
}
```

## Trend Template fail
Intent: TREND_TEMPLATE_FAIL
- eligible: `False`
- headline: NOT STAGE-2 / RS LEADER
- good_stock / setup / entry: False / False / False
- rejection: ['TREND_TEMPLATE_FAIL', 'VCP_NOT_DETECTED', 'NO_PIVOT']

```json
{
  "as_of_date": "2021-01-27",
  "atr": 0.8500000000000055,
  "base_depth_pct": null,
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
  "config_hash": "a1b0b5651651c768",
  "contraction_count": 0,
  "contraction_dates": [],
  "contraction_depths": [],
  "contraction_durations": [],
  "data_timestamp": "2021-01-27",
  "distance_from_pivot_pct": null,
  "dry_up_ratio": null,
  "eligibility_version": "sepa-001.v1",
  "eligible": false,
  "entry_rejection": "NO_PIVOT",
  "entry_valid": false,
  "evidence": {
    "buy_zone_above_pct": 1.5,
    "near_sepa": false,
    "pit": {
      "ca_complete": true,
      "ca_note": "",
      "universe_complete": true,
      "universe_note": ""
    },
    "rs_injected": true,
    "vcp_evidence": {
      "lookback": 120,
      "min_reversal_pct": 2.5,
      "swing_highs": 1,
      "swing_lows": 0
    }
  },
  "extended": false,
  "final_contraction_pct": null,
  "good_entry": false,
  "good_setup": false,
  "good_stock": false,
  "headline": "NOT STAGE-2 / RS LEADER",
  "levels": {
    "above_low_pct": 0.43,
    "below_high_pct": 54.55,
    "high_52w": 207.8,
    "low_52w": 94.05,
    "price": 94.45,
    "sma150": 127.975,
    "sma200": 139.225,
    "sma200_prev": 148.675,
    "sma50": 105.475
  },
  "measured_move": null,
  "pit_safe": true,
  "pivot": null,
  "pivot_date": null,
  "pivot_type": null,
  "price": 94.45,
  "proposed_entry": null,
  "reasons": [
    "Trend template not 8/8: price_gt_150_200, sma150_gt_200, sma200_rising, sma50_leads, price_gt_sma50, off_52w_low, near_52w_high",
    "NO_SWING_STRUCTURE",
    "No structurally valid pivot \u2014 none manufactured."
  ],
  "rejection_codes": [
    "TREND_TEMPLATE_FAIL",
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
  "rs_percentile": 90.0,
  "rs_score": null,
  "rs_threshold": 70.0,
  "setup_fail_reasons": [
    "NO_SWING_STRUCTURE"
  ],
  "setup_quality": null,
  "setup_type": "",
  "stop_atr_multiple": null,
  "stop_basis": null,
  "stop_distance_pct": null,
  "stop_ok": false,
  "structural_stop": null,
  "structure_pass": false,
  "symbol": "DOWN",
  "tightness": null,
  "trend_passed": 1,
  "trend_rules": [
    {
      "detail": "Close 94.45 vs SMA150 127.97 / SMA200 139.22",
      "id": "price_gt_150_200",
      "passed": false,
      "values": {
        "price": 94.45,
        "sma150": 127.975,
        "sma200": 139.225
      }
    },
    {
      "detail": "SMA150 127.97 vs SMA200 139.22",
      "id": "sma150_gt_200",
      "passed": false,
      "values": {
        "sma150": 127.975,
        "sma200": 139.225
      }
    },
    {
      "detail": "SMA200 139.22 vs 148.68 (21 sessions earlier)",
      "id": "sma200_rising",
      "passed": false,
      "values": {
        "sma200": 139.225,
        "sma200_prev": 148.675
      }
    },
    {
      "detail": "SMA50 105.47 vs SMA150 127.97 / SMA200 139.22",
      "id": "sma50_leads",
      "passed": false,
      "values": {
        "sma150": 127.975,
        "sma200": 139.225,
        "sma50": 105.475
      }
    },
    {
      "detail": "Close 94.45 vs SMA50 105.47",
      "id": "price_gt_sma50",
      "passed": false,
      "values": {
        "price": 94.45,
        "sma50": 105.475
      }
    },
    {
      "detail": "0.4% above 52-week low (need \u226530%)",
      "id": "off_52w_low",
      "passed": false,
      "values": {
        "above_low_pct": 0.4253056884635953,
        "low_52w": 94.05,
        "threshold": 30.0
      }
    },
    {
      "detail": "54.5% below 52-week high (need \u226425%)",
      "id": "near_52w_high",
      "passed": false,
      "values": {
        "below_high_pct": 54.54764196342637,
        "high_52w": 207.8,
        "threshold": 25.0
      }
    },
    {
      "detail": "RS percentile 90.0 (need \u226570)",
      "id": "rs_percentile",
      "passed": true,
      "values": {
        "rs_percentile": 90.0,
        "threshold": 70.0
      }
    }
  ],
  "trend_template_pass": false,
  "trend_total": 8,
  "universe_complete": true,
  "universe_version": "pit_universe",
  "vcp_detected": false,
  "vol_final": null,
  "vol_first": null,
  "vol_recent_vs_base": null
}
```

## RS fail
Intent: RS_FAIL
- eligible: `False`
- headline: NOT STAGE-2 / RS LEADER
- good_stock / setup / entry: False / False / False
- rejection: ['TREND_TEMPLATE_FAIL', 'RS_FAIL', 'VCP_NOT_DETECTED', 'NO_PIVOT']

```json
{
  "as_of_date": "2021-01-27",
  "atr": 0.950000000000007,
  "base_depth_pct": null,
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
  "config_hash": "a1b0b5651651c768",
  "contraction_count": 0,
  "contraction_dates": [],
  "contraction_depths": [],
  "contraction_durations": [],
  "data_timestamp": "2021-01-27",
  "distance_from_pivot_pct": null,
  "dry_up_ratio": null,
  "eligibility_version": "sepa-001.v1",
  "eligible": false,
  "entry_rejection": "NO_PIVOT",
  "entry_valid": false,
  "evidence": {
    "buy_zone_above_pct": 1.5,
    "near_sepa": true,
    "pit": {
      "ca_complete": true,
      "ca_note": "",
      "universe_complete": true,
      "universe_note": ""
    },
    "rs_injected": true,
    "vcp_evidence": {
      "lookback": 120,
      "min_reversal_pct": 2.5,
      "swing_highs": 0,
      "swing_lows": 0
    }
  },
  "extended": false,
  "final_contraction_pct": null,
  "good_entry": false,
  "good_setup": false,
  "good_stock": false,
  "headline": "NOT STAGE-2 / RS LEADER",
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
  "pit_safe": true,
  "pivot": null,
  "pivot_date": null,
  "pivot_type": null,
  "price": 233.45000000000002,
  "proposed_entry": null,
  "reasons": [
    "Trend template not 8/8: rs_percentile",
    "NO_SWING_STRUCTURE",
    "No structurally valid pivot \u2014 none manufactured."
  ],
  "rejection_codes": [
    "TREND_TEMPLATE_FAIL",
    "RS_FAIL",
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
  "rs_pass": false,
  "rs_percentile": 55.0,
  "rs_score": null,
  "rs_threshold": 70.0,
  "setup_fail_reasons": [
    "NO_SWING_STRUCTURE"
  ],
  "setup_quality": null,
  "setup_type": "",
  "stop_atr_multiple": null,
  "stop_basis": null,
  "stop_distance_pct": null,
  "stop_ok": false,
  "structural_stop": null,
  "structure_pass": true,
  "symbol": "LAG",
  "tightness": null,
  "trend_passed": 7,
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
  "vcp_detected": false,
  "vol_final": null,
  "vol_first": null,
  "vol_recent_vs_base": null
}
```

## False VCP grind
Intent: VCP_NOT_DETECTED
- eligible: `False`
- headline: GOOD STOCK — SETUP NOT STRUCTURAL
- good_stock / setup / entry: True / False / False
- rejection: ['VCP_NOT_DETECTED', 'NO_PIVOT']

```json
{
  "as_of_date": "2021-01-27",
  "atr": 0.950000000000007,
  "base_depth_pct": null,
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
  "config_hash": "a1b0b5651651c768",
  "contraction_count": 0,
  "contraction_dates": [],
  "contraction_depths": [],
  "contraction_durations": [],
  "data_timestamp": "2021-01-27",
  "distance_from_pivot_pct": null,
  "dry_up_ratio": null,
  "eligibility_version": "sepa-001.v1",
  "eligible": false,
  "entry_rejection": "NO_PIVOT",
  "entry_valid": false,
  "evidence": {
    "buy_zone_above_pct": 1.5,
    "near_sepa": false,
    "pit": {
      "ca_complete": true,
      "ca_note": "",
      "universe_complete": true,
      "universe_note": ""
    },
    "rs_injected": true,
    "vcp_evidence": {
      "lookback": 120,
      "min_reversal_pct": 2.5,
      "swing_highs": 0,
      "swing_lows": 0
    }
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
  "pit_safe": true,
  "pivot": null,
  "pivot_date": null,
  "pivot_type": null,
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
  "rs_percentile": 90.0,
  "rs_score": null,
  "rs_threshold": 70.0,
  "setup_fail_reasons": [
    "NO_SWING_STRUCTURE"
  ],
  "setup_quality": null,
  "setup_type": "",
  "stop_atr_multiple": null,
  "stop_basis": null,
  "stop_distance_pct": null,
  "stop_ok": false,
  "structural_stop": null,
  "structure_pass": true,
  "symbol": "COIL",
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
      "detail": "RS percentile 90.0 (need \u226570)",
      "id": "rs_percentile",
      "passed": true,
      "values": {
        "rs_percentile": 90.0,
        "threshold": 70.0
      }
    }
  ],
  "trend_template_pass": true,
  "trend_total": 8,
  "universe_complete": true,
  "universe_version": "pit_universe",
  "vcp_detected": false,
  "vol_final": null,
  "vol_first": null,
  "vol_recent_vs_base": null
}
```

## VCP extended 7%
Intent: NO TRADE INVALID ENTRY
- eligible: `False`
- headline: NO TRADE — INVALID ENTRY
- good_stock / setup / entry: True / True / False
- rejection: ['ENTRY_EXTENDED']

```json
{
  "as_of_date": "2021-03-16",
  "atr": 2.818028571428582,
  "base_depth_pct": 14.188,
  "benchmark_rs": {
    "available": false,
    "benchmark_pct": null,
    "excess_pp": null,
    "label": "UNKNOWN",
    "lookback": 63,
    "note": "Need ~3 months of official stock and Nifty history.",
    "stock_pct": null
  },
  "buy_zone_high": 213.0485,
  "buy_zone_low": 209.3753,
  "ca_complete": true,
  "config_hash": "a1b0b5651651c768",
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
  "distance_from_pivot_pct": 6.7961,
  "dry_up_ratio": 0.2478,
  "eligibility_version": "sepa-001.v1",
  "eligible": false,
  "entry_rejection": "ENTRY_EXTENDED",
  "entry_valid": false,
  "evidence": {
    "buy_zone_above_pct": 1.5,
    "near_sepa": false,
    "pit": {
      "ca_complete": true,
      "ca_note": "",
      "universe_complete": true,
      "universe_note": ""
    },
    "rs_injected": true,
    "vcp_evidence": {
      "lookback": 120,
      "min_reversal_pct": 2.5,
      "raw_contraction_count": 3,
      "swing_highs": 3,
      "swing_lows": 3
    }
  },
  "extended": true,
  "final_contraction_pct": 4.161,
  "good_entry": false,
  "good_setup": true,
  "good_stock": true,
  "headline": "NO TRADE \u2014 INVALID ENTRY",
  "levels": {
    "above_low_pct": 102.68,
    "below_high_pct": 0.13,
    "high_52w": 224.465,
    "low_52w": 110.6,
    "price": 224.165,
    "sma150": 191.6408,
    "sma200": 181.0431,
    "sma200_prev": 172.8248,
    "sma50": 201.6272
  },
  "measured_move": 14.188,
  "pit_safe": true,
  "pivot": 209.9,
  "pivot_date": "2020-12-30",
  "pivot_type": "vcp_resistance_swing_high",
  "price": 224.16500000000002,
  "proposed_entry": 224.165,
  "reasons": [
    "NO TRADE \u2014 INVALID ENTRY: price 224.16500000000002 vs pivot 209.9 (6.7961% above)."
  ],
  "rejection_codes": [
    "ENTRY_EXTENDED"
  ],
  "research_grade": false,
  "resistance": {
    "kind": "measured_move_from_first_contraction",
    "price": 239.6806
  },
  "reward_price": 239.6806,
  "reward_risk": 0.672,
  "reward_status": "MEASURED_MOVE",
  "risk_r": null,
  "rs_components": {},
  "rs_pass": true,
  "rs_percentile": 97.0,
  "rs_score": null,
  "rs_threshold": 70.0,
  "setup_fail_reasons": [],
  "setup_quality": 89.59,
  "setup_type": "VCP",
  "stop_atr_multiple": 8.1954,
  "stop_basis": "final_contraction_low",
  "stop_distance_pct": 10.3027,
  "stop_ok": false,
  "structural_stop": 201.07,
  "structure_pass": true,
  "symbol": "CHASE",
  "tightness": 0.2933,
  "trend_passed": 8,
  "trend_rules": [
    {
      "detail": "Close 224.17 vs SMA150 191.64 / SMA200 181.04",
      "id": "price_gt_150_200",
      "passed": true,
      "values": {
        "price": 224.16500000000002,
        "sma150": 191.64081650000003,
        "sma200": 181.04311237500002
      }
    },
    {
      "detail": "SMA150 191.64 vs SMA200 181.04",
      "id": "sma150_gt_200",
      "passed": true,
      "values": {
        "sma150": 191.64081650000003,
        "sma200": 181.04311237500002
      }
    },
    {
      "detail": "SMA200 181.04 vs 172.82 (21 sessions earlier)",
      "id": "sma200_rising",
      "passed": true,
      "values": {
        "sma200": 181.04311237500002,
        "sma200_prev": 172.824829875
      }
    },
    {
      "detail": "SMA50 201.63 vs SMA150 191.64 / SMA200 181.04",
      "id": "sma50_leads",
      "passed": true,
      "values": {
        "sma150": 191.64081650000003,
        "sma200": 181.04311237500002,
        "sma50": 201.62719950000007
      }
    },
    {
      "detail": "Close 224.17 vs SMA50 201.63",
      "id": "price_gt_sma50",
      "passed": true,
      "values": {
        "price": 224.16500000000002,
        "sma50": 201.62719950000007
      }
    },
    {
      "detail": "102.7% above 52-week low (need \u226530%)",
      "id": "off_52w_low",
      "passed": true,
      "values": {
        "above_low_pct": 102.68083182640146,
        "low_52w": 110.6,
        "threshold": 30.0
      }
    },
    {
      "detail": "0.1% below 52-week high (need \u226425%)",
      "id": "near_52w_high",
      "passed": true,
      "values": {
        "below_high_pct": 0.13365112601073648,
        "high_52w": 224.46500000000003,
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
  "vol_final": 93333.33,
  "vol_first": 376666.67,
  "vol_recent_vs_base": 0.3098
}
```

## Good stock, bad entry
Intent: good_stock True eligible False
- eligible: `False`
- headline: NO TRADE — INVALID ENTRY
- good_stock / setup / entry: True / True / False
- rejection: ['ENTRY_EXTENDED']

```json
{
  "as_of_date": "2021-03-16",
  "atr": 2.818028571428582,
  "base_depth_pct": 14.188,
  "benchmark_rs": {
    "available": false,
    "benchmark_pct": null,
    "excess_pp": null,
    "label": "UNKNOWN",
    "lookback": 63,
    "note": "Need ~3 months of official stock and Nifty history.",
    "stock_pct": null
  },
  "buy_zone_high": 213.0485,
  "buy_zone_low": 209.3753,
  "ca_complete": true,
  "config_hash": "a1b0b5651651c768",
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
  "distance_from_pivot_pct": 6.7961,
  "dry_up_ratio": 0.2478,
  "eligibility_version": "sepa-001.v1",
  "eligible": false,
  "entry_rejection": "ENTRY_EXTENDED",
  "entry_valid": false,
  "evidence": {
    "buy_zone_above_pct": 1.5,
    "near_sepa": false,
    "pit": {
      "ca_complete": true,
      "ca_note": "",
      "universe_complete": true,
      "universe_note": ""
    },
    "rs_injected": true,
    "vcp_evidence": {
      "lookback": 120,
      "min_reversal_pct": 2.5,
      "raw_contraction_count": 3,
      "swing_highs": 3,
      "swing_lows": 3
    }
  },
  "extended": true,
  "final_contraction_pct": 4.161,
  "good_entry": false,
  "good_setup": true,
  "good_stock": true,
  "headline": "NO TRADE \u2014 INVALID ENTRY",
  "levels": {
    "above_low_pct": 102.68,
    "below_high_pct": 0.13,
    "high_52w": 224.465,
    "low_52w": 110.6,
    "price": 224.165,
    "sma150": 191.6408,
    "sma200": 181.0431,
    "sma200_prev": 172.8248,
    "sma50": 201.6272
  },
  "measured_move": 14.188,
  "pit_safe": true,
  "pivot": 209.9,
  "pivot_date": "2020-12-30",
  "pivot_type": "vcp_resistance_swing_high",
  "price": 224.16500000000002,
  "proposed_entry": 224.165,
  "reasons": [
    "NO TRADE \u2014 INVALID ENTRY: price 224.16500000000002 vs pivot 209.9 (6.7961% above)."
  ],
  "rejection_codes": [
    "ENTRY_EXTENDED"
  ],
  "research_grade": false,
  "resistance": {
    "kind": "measured_move_from_first_contraction",
    "price": 239.6806
  },
  "reward_price": 239.6806,
  "reward_risk": 0.672,
  "reward_status": "MEASURED_MOVE",
  "risk_r": null,
  "rs_components": {},
  "rs_pass": true,
  "rs_percentile": 96.0,
  "rs_score": null,
  "rs_threshold": 70.0,
  "setup_fail_reasons": [],
  "setup_quality": 89.59,
  "setup_type": "VCP",
  "stop_atr_multiple": 8.1954,
  "stop_basis": "final_contraction_low",
  "stop_distance_pct": 10.3027,
  "stop_ok": false,
  "structural_stop": 201.07,
  "structure_pass": true,
  "symbol": "GOODBAD",
  "tightness": 0.2933,
  "trend_passed": 8,
  "trend_rules": [
    {
      "detail": "Close 224.17 vs SMA150 191.64 / SMA200 181.04",
      "id": "price_gt_150_200",
      "passed": true,
      "values": {
        "price": 224.16500000000002,
        "sma150": 191.64081650000003,
        "sma200": 181.04311237500002
      }
    },
    {
      "detail": "SMA150 191.64 vs SMA200 181.04",
      "id": "sma150_gt_200",
      "passed": true,
      "values": {
        "sma150": 191.64081650000003,
        "sma200": 181.04311237500002
      }
    },
    {
      "detail": "SMA200 181.04 vs 172.82 (21 sessions earlier)",
      "id": "sma200_rising",
      "passed": true,
      "values": {
        "sma200": 181.04311237500002,
        "sma200_prev": 172.824829875
      }
    },
    {
      "detail": "SMA50 201.63 vs SMA150 191.64 / SMA200 181.04",
      "id": "sma50_leads",
      "passed": true,
      "values": {
        "sma150": 191.64081650000003,
        "sma200": 181.04311237500002,
        "sma50": 201.62719950000007
      }
    },
    {
      "detail": "Close 224.17 vs SMA50 201.63",
      "id": "price_gt_sma50",
      "passed": true,
      "values": {
        "price": 224.16500000000002,
        "sma50": 201.62719950000007
      }
    },
    {
      "detail": "102.7% above 52-week low (need \u226530%)",
      "id": "off_52w_low",
      "passed": true,
      "values": {
        "above_low_pct": 102.68083182640146,
        "low_52w": 110.6,
        "threshold": 30.0
      }
    },
    {
      "detail": "0.1% below 52-week high (need \u226425%)",
      "id": "near_52w_high",
      "passed": true,
      "values": {
        "below_high_pct": 0.13365112601073648,
        "high_52w": 224.46500000000003,
        "threshold": 25.0
      }
    },
    {
      "detail": "RS percentile 96.0 (need \u226570)",
      "id": "rs_percentile",
      "passed": true,
      "values": {
        "rs_percentile": 96.0,
        "threshold": 70.0
      }
    }
  ],
  "trend_template_pass": true,
  "trend_total": 8,
  "universe_complete": true,
  "universe_version": "pit_universe",
  "vcp_detected": true,
  "vol_final": 93333.33,
  "vol_first": 376666.67,
  "vol_recent_vs_base": 0.3098
}
```

## Wide structural stop
Intent: WIDE_STRUCTURAL_STOP
- eligible: `False`
- headline: NOT STAGE-2 / RS LEADER
- good_stock / setup / entry: False / False / False
- rejection: ['TREND_TEMPLATE_FAIL', 'VCP_NOT_DETECTED', 'WIDE_STRUCTURAL_STOP']

```json
{
  "as_of_date": "2021-03-16",
  "atr": 15.816585714285724,
  "base_depth_pct": 57.106,
  "benchmark_rs": {
    "available": false,
    "benchmark_pct": null,
    "excess_pp": null,
    "label": "UNKNOWN",
    "lookback": 63,
    "note": "Need ~3 months of official stock and Nifty history.",
    "stock_pct": null
  },
  "buy_zone_high": 213.0485,
  "buy_zone_low": 209.3753,
  "ca_complete": true,
  "config_hash": "a1b0b5651651c768",
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
  "distance_from_pivot_pct": -0.1906,
  "dry_up_ratio": 0.2478,
  "eligibility_version": "sepa-001.v1",
  "eligible": false,
  "entry_rejection": "WIDE_STRUCTURAL_STOP",
  "entry_valid": true,
  "evidence": {
    "buy_zone_above_pct": 1.5,
    "near_sepa": true,
    "pit": {
      "ca_complete": true,
      "ca_note": "",
      "universe_complete": true,
      "universe_note": ""
    },
    "rs_injected": true,
    "vcp_evidence": {
      "lookback": 120,
      "min_reversal_pct": 2.5,
      "raw_contraction_count": 3,
      "swing_highs": 3,
      "swing_lows": 3
    }
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
  "pit_safe": true,
  "pivot": 209.9,
  "pivot_date": "2020-12-30",
  "pivot_type": "vcp_resistance_swing_high",
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
    "price": 329.7655
  },
  "reward_price": 329.7655,
  "reward_risk": 1.103,
  "reward_status": "MEASURED_MOVE",
  "risk_r": null,
  "rs_components": {},
  "rs_pass": true,
  "rs_percentile": 90.0,
  "rs_score": null,
  "rs_threshold": 70.0,
  "setup_fail_reasons": [
    "NOT_TIGHTENING",
    "FINAL_CONTRACTION_LOOSE",
    "BASE_TOO_DEEP"
  ],
  "setup_quality": 50.0,
  "setup_type": "",
  "stop_atr_multiple": 6.8909,
  "stop_basis": "final_contraction_low",
  "stop_distance_pct": 52.0239,
  "stop_ok": false,
  "structural_stop": 100.51,
  "structure_pass": false,
  "symbol": "WIDE",
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
      "detail": "RS percentile 90.0 (need \u226570)",
      "id": "rs_percentile",
      "passed": true,
      "values": {
        "rs_percentile": 90.0,
        "threshold": 70.0
      }
    }
  ],
  "trend_template_pass": false,
  "trend_total": 8,
  "universe_complete": true,
  "universe_version": "pit_universe",
  "vcp_detected": false,
  "vol_final": 93333.33,
  "vol_first": 376666.67,
  "vol_recent_vs_base": 0.3098
}
```

## Eligible setup
Intent: eligible True
- eligible: `True`
- headline: ELIGIBLE — stock + setup + entry
- good_stock / setup / entry: True / True / True
- rejection: []

```json
{
  "as_of_date": "2021-03-16",
  "atr": 1.8303857142857234,
  "base_depth_pct": 14.188,
  "benchmark_rs": {
    "available": false,
    "benchmark_pct": null,
    "excess_pp": null,
    "label": "UNKNOWN",
    "lookback": 63,
    "note": "Need ~3 months of official stock and Nifty history.",
    "stock_pct": null
  },
  "buy_zone_high": 213.0485,
  "buy_zone_low": 209.3753,
  "ca_complete": true,
  "config_hash": "a1b0b5651651c768",
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
  "distance_from_pivot_pct": 0.2087,
  "dry_up_ratio": 0.2478,
  "eligibility_version": "sepa-001.v1",
  "eligible": true,
  "entry_rejection": null,
  "entry_valid": true,
  "evidence": {
    "buy_zone_above_pct": 1.5,
    "near_sepa": false,
    "pit": {
      "ca_complete": true,
      "ca_note": "",
      "universe_complete": true,
      "universe_note": ""
    },
    "rs_injected": true,
    "vcp_evidence": {
      "lookback": 120,
      "min_reversal_pct": 2.5,
      "raw_contraction_count": 3,
      "swing_highs": 3,
      "swing_lows": 3
    }
  },
  "extended": false,
  "final_contraction_pct": 4.161,
  "good_entry": true,
  "good_setup": true,
  "good_stock": true,
  "headline": "ELIGIBLE \u2014 stock + setup + entry",
  "levels": {
    "above_low_pct": 90.18,
    "below_high_pct": 0.14,
    "high_52w": 210.638,
    "low_52w": 110.6,
    "price": 210.338,
    "sma150": 191.1799,
    "sma200": 180.6974,
    "sma200_prev": 172.8248,
    "sma50": 200.2445
  },
  "measured_move": 14.188,
  "pit_safe": true,
  "pivot": 209.9,
  "pivot_date": "2020-12-30",
  "pivot_type": "vcp_resistance_swing_high",
  "price": 210.338,
  "proposed_entry": 210.338,
  "reasons": [],
  "rejection_codes": [],
  "research_grade": true,
  "resistance": {
    "kind": "measured_move_from_first_contraction",
    "price": 239.6806
  },
  "reward_price": 239.6806,
  "reward_risk": 3.166,
  "reward_status": "MEASURED_MOVE",
  "risk_r": 1.0,
  "rs_components": {},
  "rs_pass": true,
  "rs_percentile": 88.0,
  "rs_score": null,
  "rs_threshold": 70.0,
  "setup_fail_reasons": [],
  "setup_quality": 89.59,
  "setup_type": "VCP",
  "stop_atr_multiple": 5.0634,
  "stop_basis": "final_contraction_low",
  "stop_distance_pct": 4.4062,
  "stop_ok": true,
  "structural_stop": 201.07,
  "structure_pass": true,
  "symbol": "ELIG",
  "tightness": 0.2933,
  "trend_passed": 8,
  "trend_rules": [
    {
      "detail": "Close 210.34 vs SMA150 191.18 / SMA200 180.70",
      "id": "price_gt_150_200",
      "passed": true,
      "values": {
        "price": 210.338,
        "sma150": 191.1799165,
        "sma200": 180.69743737500002
      }
    },
    {
      "detail": "SMA150 191.18 vs SMA200 180.70",
      "id": "sma150_gt_200",
      "passed": true,
      "values": {
        "sma150": 191.1799165,
        "sma200": 180.69743737500002
      }
    },
    {
      "detail": "SMA200 180.70 vs 172.82 (21 sessions earlier)",
      "id": "sma200_rising",
      "passed": true,
      "values": {
        "sma200": 180.69743737500002,
        "sma200_prev": 172.824829875
      }
    },
    {
      "detail": "SMA50 200.24 vs SMA150 191.18 / SMA200 180.70",
      "id": "sma50_leads",
      "passed": true,
      "values": {
        "sma150": 191.1799165,
        "sma200": 180.69743737500002,
        "sma50": 200.24449950000002
      }
    },
    {
      "detail": "Close 210.34 vs SMA50 200.24",
      "id": "price_gt_sma50",
      "passed": true,
      "values": {
        "price": 210.338,
        "sma50": 200.24449950000002
      }
    },
    {
      "detail": "90.2% above 52-week low (need \u226530%)",
      "id": "off_52w_low",
      "passed": true,
      "values": {
        "above_low_pct": 90.17902350813745,
        "low_52w": 110.6,
        "threshold": 30.0
      }
    },
    {
      "detail": "0.1% below 52-week high (need \u226425%)",
      "id": "near_52w_high",
      "passed": true,
      "values": {
        "below_high_pct": 0.14242444383255615,
        "high_52w": 210.638,
        "threshold": 25.0
      }
    },
    {
      "detail": "RS percentile 88.0 (need \u226570)",
      "id": "rs_percentile",
      "passed": true,
      "values": {
        "rs_percentile": 88.0,
        "threshold": 70.0
      }
    }
  ],
  "trend_template_pass": true,
  "trend_total": 8,
  "universe_complete": true,
  "universe_version": "pit_universe",
  "vcp_detected": true,
  "vol_final": 93333.33,
  "vol_first": 376666.67,
  "vol_recent_vs_base": 0.3098
}
```
