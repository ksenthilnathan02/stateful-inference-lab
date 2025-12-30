# app/cache.py

import time
from collections import OrderedDict


class LRUWithTTLCache:
    """
    Hybrid cache combining:
    - LRU eviction (memory bound)
    - TTL expiration (correctness bound)

    Designed to be dynamically controlled by an agent.
    """

    def __init__(self, max_size=256, ttl_seconds=30):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.enabled = True  # 👈 Agent-controlled switch

        # key -> (value, timestamp)
        self.store = OrderedDict()

        # Metrics
        self.hits = 0
        self.misses = 0
        self.expired = 0
        self.evictions = 0

    def get(self, key):
        """
        Returns cached value if present, enabled, and not expired.
        """
        if not self.enabled:
            self.misses += 1
            return None

        if key not in self.store:
            self.misses += 1
            return None

        value, ts = self.store[key]

        # TTL expiration
        if time.time() - ts > self.ttl:
            del self.store[key]
            self.expired += 1
            self.misses += 1
            return None

        # LRU update
        self.store.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key, value):
        """
        Inserts or updates a cache entry.
        """
        if not self.enabled:
            return

        self.store[key] = (value, time.time())
        self.store.move_to_end(key)

        # LRU eviction
        if len(self.store) > self.max_size:
            self.store.popitem(last=False)
            self.evictions += 1

    def clear(self):
        """
        Clears cache entries and metrics.
        """
        self.store.clear()
        self.hits = 0
        self.misses = 0
        self.expired = 0
        self.evictions = 0

    def stats(self):
        """
        Cache observability metrics.
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "policy": "LRU+TTL",
            "enabled": self.enabled,
            "size": len(self.store),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "expired": self.expired,
            "evictions": self.evictions,
            "hit_rate": round(hit_rate, 3),
        }
