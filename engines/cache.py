"""In-memory cache with TTL support."""

import time

_cache = {}
CACHE_TTL = 600  # 10 minutes — matches prefetch interval


def get_cached(key, ttl=None):
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < (ttl or CACHE_TTL):
            return data
    return None


def set_cache(key, data):
    _cache[key] = (data, time.time())


def clear_cache():
    _cache.clear()
