"""Day 29.07 — Redis cache-aside stub for prediction caching
Run time: ~10-15 minutes

- Demonstrates cache-aside pattern with redis-py if installed.
  Prints pseudocode otherwise.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Helps static analyzers when redis isn't installed in the analysis env
    # import under a different name to avoid redefinition warnings
    import redis as _redis  # type: ignore  # noqa: F401

try:
    import redis  # type: ignore
    has_redis = True
except ImportError:
    has_redis = False

import json

CACHE_TTL = 60  # seconds


def predict_heavy(x):
    # placeholder heavy compute
    return {"result": x[::-1]}


if __name__ == "__main__":
    key = "pred:hello"

    if not has_redis:
        print("redis not installed. Pseudocode:")
        print("- r = redis.Redis(...)")
        print("- v = r.get(key)")
        print("  if v: return json.loads(v)")
        print(
            "  else: compute; r.set(key, json.dumps(out), ex=CACHE_TTL)"
        )
    else:
        r = redis.Redis()
        v = r.get(key)
        if v:
            print("Cache hit:", json.loads(v))
        else:
            out = predict_heavy("hello")
            r.set(key, json.dumps(out), ex=CACHE_TTL)
            print("Cache miss — computed and stored:", out)

    # Exercises:
    # - Implement a cache key function based on model name + input hash.
    # - Add early TTL refresh (cache warming) when TTL < threshold.
