import sqlite3
from API.config import DATABASE_PATH

def get_db():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()