import psycopg
from Utils.db_config import DB_CONFIG

try:

    conn = psycopg.connect(**DB_CONFIG)

    print("PostgreSQL connection successful!")

    conn.close()

except Exception as e:

    print("Connection failed:")
    print(e)