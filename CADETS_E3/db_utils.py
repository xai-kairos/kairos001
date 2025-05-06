"""
db_utils.py
===========

Light-weight utilities for interacting with the Postgres instance used in the
CADETS / KAIROS code-base.  Import and use either the context manager:

    >>> from db_utils import PostgresDB
    >>> with PostgresDB() as db:
    ...     tables = db.list_tables()
    ...     rows = db.execute("SELECT COUNT(*) FROM event_table")

or the functional helpers:

    >>> from db_utils import list_tables, run_sql
    >>> print(list_tables())
    >>> print(run_sql("SELECT * FROM file_node_table LIMIT 5;"))
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Any, Iterable, List, Tuple

# Re-use the existing constants
from config import DATABASE, USER, PASSWORD, HOST, PORT


###############################################################################
# Internal connection helper
###############################################################################
def _connect():
    """
    Returns (conn, cur) where cur uses RealDictCursor so results come back
    as dictionaries instead of tuples.
    """
    conn = psycopg2.connect(
        dbname=DATABASE,
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT
    )
    cur = conn.cursor(cursor_factory=RealDictCursor)
    return conn, cur


###############################################################################
# Public OO API
###############################################################################
class PostgresDB:
    """
    Context-manager wrapper around a single Postgres connection.

        with PostgresDB() as db:
            print(db.list_tables())
            rows = db.execute("SELECT * FROM event_table WHERE _id = %s", (42,))
    """

    def __init__(self):
        self.conn, self.cur = _connect()

    # ---- Context-manager plumbing -----------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc:
            self.conn.rollback()
        else:
            self.conn.commit()
        self.cur.close()
        self.conn.close()

    # ---- Convenience methods ----------------------------------------------
    def list_tables(self) -> List[str]:
        """Return all *regular* (BASE TABLE) names in the current database."""
        self.cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type   = 'BASE TABLE'
            ORDER BY table_name;
            """
        )
        return [row["table_name"] for row in self.cur.fetchall()]

    def execute(
        self,
        sql: str,
        params: Tuple[Any, ...] | List[Any] | None = None,
        fetch: bool = True,
    ) -> List[dict] | None:
        """
        Run *sql* with optional *params*.  If `fetch` is True (default) the
        method returns all rows (`list[dict]`), otherwise it returns None.

        Example:
            rows = db.execute("SELECT * FROM event_table WHERE _id = %s", (1,))
        """
        self.cur.execute(sql, params)
        if fetch:
            return self.cur.fetchall()
        else:
            return self.cur.rowcount 


###############################################################################
# Functional wrappers (one-off usage)
###############################################################################
@contextmanager
def _session():
    conn, cur = _connect()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def list_tables() -> List[str]:
    """Functional façade for PostgresDB.list_tables()."""
    with _session() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type   = 'BASE TABLE'
            ORDER BY table_name;
            """
        )
        return [row["table_name"] for row in cur.fetchall()]


def run_sql(
    sql: str,
    params: Tuple[Any, ...] | List[Any] | None = None,
    fetch: bool = True,
) -> List[dict] | None:
    """
    Execute *sql* once and close the connection.  Mirrors PostgresDB.execute().
    """
    with _session() as cur:
        cur.execute(sql, params)
        if fetch:
            return cur.fetchall()
        else:
            return cur.rowcount
    