import psycopg
from Utils.db_config import DB_CONFIG

def get_db_pg():
    with psycopg.connect(**DB_CONFIG) as conn:
        yield conn