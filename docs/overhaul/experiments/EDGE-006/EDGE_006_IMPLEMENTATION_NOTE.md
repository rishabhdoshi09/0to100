# EDGE-006 — Implementation Note

Written before results. Last new primary EDGE under this mandate.

ADV is `FastInvestable._turn[i][j]` — rolling 20-session mean of close×volume, bars ≤ T. Already used as a floor, never as a ranker.

Score = ADV. Top 20 highest.
