"""Day 28.06 — DB integration stub: PostgreSQL and Redis connection examples
Run time: ~10-15 minutes

- Safe stubs that demonstrate connection patterns without requiring DB server
"""

try:
    import psycopg2
    has_pg = True
except Exception:
    has_pg = False

try:
    import redis
    has_redis = True
except Exception:
    has_redis = False

if __name__ == '__main__':
    if not has_pg:
        print('psycopg2 not installed. To test Postgres integration, install psycopg2-binary')
    else:
        print('psycopg2 installed — example connect string:')
        print("conn = psycopg2.connect(dbname='db', user='user', password='pass', host='localhost')")

    if not has_redis:
        print('redis-py not installed. To test Redis, install redis package.')
    else:
        print('Redis installed — example: r = redis.Redis(host="localhost", port=6379)')

    # Exercises:
    # - Write a small function to cache predictions in Redis with TTL.
    # - Sketch a migration process for Postgres schema changes.