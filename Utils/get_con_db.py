import psycopg
from Utils.db_config import DB_CONFIG

def get_connection():
    return psycopg.connect(**DB_CONFIG)