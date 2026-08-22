# EDGE-004 — Results

**Classification: `REJECT`**

Net CAGR 12.55% vs EW 25.62% (excess -13.07%). Sharpe 0.556, max DD -43.10%. Worst month -20.30% (2023-01-31). Decile Spearman -0.139. D10−D1 -0.02%.

Inference: {'mean': -0.007684785408553374, 'ci': {'mean_r': -0.007684785408553374, 'ci_lower': -0.017150478914518555, 'ci_upper': 0.0012552958740025928, 'excludes_zero': False, 'block': 4.0, 'n_eff': 70.0, 'n_boot': 2000}, 'sharpe': -0.6709138501121761, 'psr': 9.626953512223187e-07, 'dsr': {'dsr': 9.626953512223187e-07, 'sr0_expected_max_null': 0.0, 'n_trials': 24, 'sharpe_variance': 0.0}, 'p_beat_ew': 0.42857142857142855}
Harness: `REJECT` — No edge — -0.01R over 70 trades. Not worth acting on.

Failures: `['net_does_not_beat_ew_or_nifty', 'deciles_not_inverse', 'losers_do_not_beat_winners']`
