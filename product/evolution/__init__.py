"""QuantTerm Evolution Engine — Champion vs Challenger PAPER policy tournament.

This package answers a different question than the rest of the learning stack.
Everywhere else in QuantTerm (conditional_evidence, learning_policy_store,
decision_ranking's evidence adjustment) improves the SCORE produced by the one
decision process QuantTerm already runs. This package instead asks whether an
ENTIRE alternative decision process -- a different, named, bounded weighting
of the same real evidence -- would have made better decisions than the one
currently in PAPER authority, using the exact same frozen point-in-time
information.

Only the current Champion policy may ever reach real PAPER execution.
Every other policy is a shadow: it freezes a decision record, is graded once
the real forward outcome exists, and never touches any book, broker, or
Telegram execution path. See product/evolution/policy_eval.py and
product/evolution/tournament.py for the enforcement boundary.

live_locked=True, live_execution_authorized=False throughout -- this package
has no broker/live code path at all, by construction (see policy_eval.py's
module docstring).
"""
