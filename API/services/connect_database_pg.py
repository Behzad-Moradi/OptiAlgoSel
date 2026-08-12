import psycopg
from Utils.db_config import DB_CONFIG

def get_db_pg():
    conn = psycopg.connect(**DB_CONFIG)

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()