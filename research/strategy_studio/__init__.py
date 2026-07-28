"""
🧪 Strategy Studio — autonomous strategy discovery + user-guided approval workflow.

RESEARCH + HUMAN-IN-THE-LOOP ONLY. The system may generate and REJECT strategies on its
own, but it must NEVER silently implement, paper-deploy, promote or live-deploy one. No
module here can place a broker order. Only a USER may approve a strategy for PAPER; LIVE
is out of scope and stays migration-locked. Synthetic fixtures verify software behaviour
and are NEVER presented as market evidence.
"""
from research.strategy_studio import spec, grammar, discovery, review, tweak, approval, wizard

__all__ = ["spec", "grammar", "discovery", "review", "tweak", "approval", "wizard"]
