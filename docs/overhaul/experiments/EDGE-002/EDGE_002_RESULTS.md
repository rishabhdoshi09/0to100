# EDGE-002 — Results

**Classification: `REJECT`**

Net CAGR 6.38% vs EW 25.62% (excess -19.23%). Sharpe 0.898, max DD -11.46%. Worst month -6.51% (2021-02-26). Decile Spearman -0.539.

Inference: {'mean': -0.015533319781834098, 'ci': {'mean_r': -0.015533319781834098, 'ci_lower': -0.030143709549623445, 'ci_upper': -0.001902853747503347, 'excludes_zero': False, 'block': 4.0, 'n_eff': 47.115599601458435, 'n_boot': 2000}, 'sharpe': -0.965416267265638, 'psr': 6.786999528921858e-11, 'dsr': {'dsr': 6.786999528921858e-11, 'sr0_expected_max_null': 0.0, 'n_trials': 48, 'sharpe_variance': 0.0}, 'p_beat_ew': 0.4}
Harness: `REJECT` — No edge — -0.02R over 70 trades. Not worth acting on.

Failures: `['net_does_not_beat_ew_or_nifty', 'deciles_not_ordered']`
