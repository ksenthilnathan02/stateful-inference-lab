# app/agent/policy.py

class CachePolicyAgent:
    """
    Simple rule-based agent that decides cache behavior
    based on system health signals.
    """

    def decide(self, metrics: dict) -> str:
        """
        Returns one of:
        - NORMAL
        - SHORT_TTL
        - DISABLE_CACHE
        """

        if metrics.get("correctness_risk", 0) > 0.8:
            return "DISABLE_CACHE"

        if metrics.get("drift_score", 0) > 0.6:
            return "SHORT_TTL"

        return "NORMAL"
