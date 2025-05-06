from db_utils import PostgresDB, list_tables, run_sql

# OO flavour
with PostgresDB() as db:
    print("Tables:", db.list_tables())
    rows = db.execute("SELECT COUNT(*) AS n FROM event_table")
    print(rows[0]["n"])

# Functional, one-off query
print("All tables:", list_tables())
result = run_sql(
    """
SELECT
  
  to_char(
    to_timestamp(timestamp_rec / 1e9),
    'DD-MM-YYYY HH24:MI:SS'
  ) AS event_time
FROM (
  SELECT timestamp_rec
  FROM event_table
) AS distinct_ts
ORDER BY timestamp_rec DESC;
    """,
    fetch=True,
)
# result is a list of dicts; grab the first row's "max_event_time"
if result:
    print("Result:", result)
else:
    print("No data returned.")
