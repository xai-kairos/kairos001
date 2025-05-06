import psycopg2
from config import *
from psycopg2 import sql


def get_database_stats():
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname=DATABASE,
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT
        )
        cur = conn.cursor()
        cur.execute("ANALYZE;")


        # Get total size of the database
        cur.execute("SELECT pg_size_pretty(pg_database_size(%s));", (DATABASE,))
        total_size = cur.fetchone()[0]
        print(f"Total size of database '{DATABASE}': {total_size}")

        # Get row count and size for each table
        query = """
            SELECT 
                table_schema || '.' || table_name AS table_name,
                pg_size_pretty(pg_total_relation_size('"' || table_schema || '"."' || table_name || '"')) AS table_size,
                pg_relation_size('"' || table_schema || '"."' || table_name || '"') AS raw_size,
                reltuples::BIGINT AS row_count
            FROM 
                information_schema.tables
            JOIN 
                pg_class ON table_name = relname
            WHERE 
                table_schema NOT IN ('pg_catalog', 'information_schema')
                AND table_type = 'BASE TABLE'
            ORDER BY 
                raw_size DESC;
        """
        cur.execute(query)
        tables = cur.fetchall()

        # Print table stats
        print("\nTable Stats:")
        print(f"{'Table Name':<40} {'Size':<15} {'Row Count':<20}")
        print('-' * 80)
        for table in tables:
            table_name, table_size, _, row_count = table
            
            # If row_count is -1, use COUNT(*) for accuracy
            if row_count == -1:
                print(f"Counting rows for {table_name} (this may take time)...")
                count_query = sql.SQL("SELECT COUNT(*) FROM {}")
                count_query = count_query.format(sql.Identifier(*table_name.split('.')))
                cur.execute(count_query)
                row_count = cur.fetchone()[0]
            
            print(f"{table_name:<40} {table_size:<15} {row_count:<20}")

        # Output first 5 rows for each table
        print("\nFirst 5 rows of each table:")
        for table in tables:
            table_name = table[0]
            try:
                # Split into schema and table name; default to public if not provided
                parts = table_name.split('.')
                if len(parts) == 2:
                    schema, tbl = parts
                else:
                    schema = 'public'
                    tbl = parts[0]

                print(f"\nTable: {table_name}")
                row_query = sql.SQL("SELECT * FROM {}.{} LIMIT 5").format(
                    sql.Identifier(schema),
                    sql.Identifier(tbl)
                )
                cur.execute(row_query)
                rows = cur.fetchall()
                # Retrieve column names
                colnames = [desc[0] for desc in cur.description]
                print(colnames)
                for row in rows:
                    print(row)
            except Exception as e:
                print(f"Could not fetch rows for {table_name}: {e}")

        # Close connection
        cur.close()
        conn.close()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    get_database_stats()
