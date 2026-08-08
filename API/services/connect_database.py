import sqlite3
from API.config import settings

def get_db():
    conn = sqlite3.connect(settings.DATABASE_PATH, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()