import psycopg2
from config import *
from psycopg2 import sql

# Database connection parameters
DB_SUPERUSER = USER         # Superuser to create the database
DB_PASSWORD = PASSWORD      # Replace with the postgres user's password
DB_HOST = HOST              # Replace with the host of the database
DB_PORT = PORT              # Replace with the port of the database

DB_NAME = DATABASE          # Replace with the name of the database to create


def drop_database():
    """
    Drop the database if it exists.
    """
    try:
        # Connect to PostgreSQL as superuser to drop the database
        conn = psycopg2.connect(
            dbname='postgres',
            user=DB_SUPERUSER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = True  # Enable autocommit for database operations
        cur = conn.cursor()

        # Check if the database exists
        cur.execute(
            "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_NAME,))
        exists = cur.fetchone()
        if exists:
            # Drop the database
            cur.execute(sql.SQL("DROP DATABASE {}").format(
                sql.Identifier(DB_NAME)))
            print(f"Database '{DB_NAME}' dropped successfully.")
        else:
            print(f"Database '{DB_NAME}' does not exist. No need to drop.")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"An error occurred while dropping the database: {e}")
        exit(1)


def create_database():
    """
    Create the database.
    """
    try:
        # Connect to PostgreSQL as superuser to create the database
        conn = psycopg2.connect(
            dbname='postgres',
            user=DB_SUPERUSER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = True  # Enable autocommit for database creation
        cur = conn.cursor()

        # Create the database
        cur.execute(sql.SQL("CREATE DATABASE {}").format(
            sql.Identifier(DB_NAME)))
        print(f"Database '{DB_NAME}' created successfully.")

        cur.close()
        conn.close()
    except psycopg2.errors.DuplicateDatabase:
        print(f"Database '{DB_NAME}' already exists. Skipping creation.")
    except Exception as e:
        print(f"An error occurred while creating the database: {e}")
        exit(1)


def create_tables():
    """
    Create the tables in the database.
    """
    try:
        # Connect to the newly created database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_SUPERUSER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = False
        cur = conn.cursor()

        # Create event_table
        cur.execute("""
            CREATE TABLE event_table (
                src_node      VARCHAR,
                src_index_id  VARCHAR,
                operation     VARCHAR,
                dst_node      VARCHAR,
                dst_index_id  VARCHAR,
                timestamp_rec BIGINT,
                _id           SERIAL
            );
        """)
        cur.execute("ALTER TABLE event_table OWNER TO postgres;")
        cur.execute("""
            CREATE UNIQUE INDEX event_table__id_uindex ON event_table (_id);
        """)
        cur.execute("""
            GRANT DELETE, INSERT, REFERENCES, SELECT, TRIGGER, TRUNCATE, UPDATE ON event_table TO postgres;
        """)
        print("Table 'event_table' created successfully.")

        # Create file_node_table
        cur.execute("""
            CREATE TABLE file_node_table (
                node_uuid VARCHAR NOT NULL,
                hash_id   VARCHAR NOT NULL,
                path      VARCHAR,
                CONSTRAINT file_node_table_pk PRIMARY KEY (node_uuid, hash_id)
            );
        """)
        cur.execute("ALTER TABLE file_node_table OWNER TO postgres;")
        print("Table 'file_node_table' created successfully.")

        # Create netflow_node_table
        cur.execute("""
            CREATE TABLE netflow_node_table (
                node_uuid VARCHAR NOT NULL,
                hash_id   VARCHAR NOT NULL,
                src_addr  VARCHAR,
                src_port  VARCHAR,
                dst_addr  VARCHAR,
                dst_port  VARCHAR,
                CONSTRAINT netflow_node_table_pk PRIMARY KEY (node_uuid, hash_id)
            );
        """)
        cur.execute("ALTER TABLE netflow_node_table OWNER TO postgres;")
        print("Table 'netflow_node_table' created successfully.")

        # Create subject_node_table
        cur.execute("""
            CREATE TABLE subject_node_table (
                node_uuid VARCHAR,
                hash_id   VARCHAR,
                exec      VARCHAR
            );
        """)
        cur.execute("ALTER TABLE subject_node_table OWNER TO postgres;")
        print("Table 'subject_node_table' created successfully.")

        # Create node2id table
        cur.execute("""
            CREATE TABLE node2id (
                hash_id   VARCHAR NOT NULL CONSTRAINT node2id_pk PRIMARY KEY,
                node_type VARCHAR,
                msg       VARCHAR,
                index_id  BIGINT
            );
        """)
        cur.execute("ALTER TABLE node2id OWNER TO postgres;")
        cur.execute("""
            CREATE UNIQUE INDEX node2id_hash_id_uindex ON node2id (hash_id);
        """)
        print("Table 'node2id' created successfully.")

        # Commit changes
        conn.commit()
        cur.close()
        conn.close()
        print("All tables created and configured successfully.")

    except Exception as e:
        print(f"An error occurred while creating tables: {e}")
        if conn:
            conn.rollback()
        if cur:
            cur.close()
        if conn:
            conn.close()
        exit(1)


if __name__ == "__main__":
    # Drop the existing database if it exists
    drop_database()

    # Create a new database
    create_database()

    # Create tables in the new database
    create_tables()
