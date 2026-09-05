"""
Minimal Postgres connection helper. Uses psycopg2 directly (no ORM) —
the schema is small and stable enough that raw SQL is simpler than
introducing SQLAlchemy for this scale.
"""
import os
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]


@contextmanager
def get_connection():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_cursor():
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur

def log_cost(call_type: str, reference: str, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO cost_log (call_type, reference, input_tokens, output_tokens, cost_usd)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (call_type, reference, input_tokens, output_tokens, cost_usd),
        )